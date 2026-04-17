package com.wq;

import dev.langchain4j.community.model.dashscope.QwenChatModel;
import dev.langchain4j.model.chat.ChatModel;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.*;
import java.lang.ref.SoftReference;
import java.util.concurrent.atomic.AtomicLong;

// 缓存条目
class CacheEntry<T> {
    private final String key;
    private final T value;
    private final long createdTime;
    private volatile long lastAccessTime;
    private final AtomicLong accessCount;
    private final int priority;
    
    public CacheEntry(String key, T value, int priority) {
        this.key = key;
        this.value = value;
        this.priority = priority;
        this.createdTime = System.currentTimeMillis();
        this.lastAccessTime = createdTime;
        this.accessCount = new AtomicLong(1);
    }
    
    public T getValue() {
        this.lastAccessTime = System.currentTimeMillis();
        this.accessCount.incrementAndGet();
        return value;
    }
    
    // getter 方法
    public String getKey() { return key; }
    public long getCreatedTime() { return createdTime; }
    public long getLastAccessTime() { return lastAccessTime; }
    public long getAccessCount() { return accessCount.get(); }
    public int getPriority() { return priority; }
    
    // 计算缓存价值
    public double getCacheValue() {
        long age = System.currentTimeMillis() - lastAccessTime;
        double timeDecay = Math.exp(-age / (24 * 60 * 60 * 1000.0)); // 按天衰减
        return priority * timeDecay * Math.log(accessCount.get() + 1);
    }
}

// 多级缓存管理器
class MultiLevelCacheManager<T> {
    // L1缓存：内存中的高速缓存
    private final Map<String, CacheEntry<T>> l1Cache;
    private final int l1MaxSize;
    
    // L2缓存：软引用缓存，内存不足时会被回收
    private final Map<String, SoftReference<CacheEntry<T>>> l2Cache;
    private final int l2MaxSize;
    
    // L3缓存：磁盘持久化缓存
    private final PersistentCache<T> l3Cache;
    
    // 访问统计
    private final AtomicLong l1Hits = new AtomicLong(0);
    private final AtomicLong l2Hits = new AtomicLong(0);
    private final AtomicLong l3Hits = new AtomicLong(0);
    private final AtomicLong misses = new AtomicLong(0);
    
    public MultiLevelCacheManager(int l1MaxSize, int l2MaxSize, String l3CacheDir) {
        this.l1MaxSize = l1MaxSize;
        this.l2MaxSize = l2MaxSize;
        this.l1Cache = new ConcurrentHashMap<>();
        this.l2Cache = new ConcurrentHashMap<>();
        this.l3Cache = new PersistentCache<>(l3CacheDir);
    }
    
    // 获取缓存值
    public T get(String key) {
        // L1缓存查找
        CacheEntry<T> entry = l1Cache.get(key);
        if (entry != null) {
            l1Hits.incrementAndGet();
            return entry.getValue();
        }
        
        // L2缓存查找
        SoftReference<CacheEntry<T>> softRef = l2Cache.get(key);
        if (softRef != null) {
            entry = softRef.get();
            if (entry != null) {
                l2Hits.incrementAndGet();
                // 提升到L1缓存
                promoteToL1(key, entry);
                return entry.getValue();
            } else {
                // 软引用已被回收，清理
                l2Cache.remove(key);
            }
        }
        
        // L3缓存查找
        T value = l3Cache.get(key);
        if (value != null) {
            l3Hits.incrementAndGet();
            // 创建新的缓存条目并提升到L1
            CacheEntry<T> newEntry = new CacheEntry<>(key, value, 1);
            promoteToL1(key, newEntry);
            return value;
        }
        
        misses.incrementAndGet();
        return null;
    }
    
    // 存储到缓存
    public void put(String key, T value, int priority) {
        CacheEntry<T> entry = new CacheEntry<>(key, value, priority);
        
        // 直接存储到L1缓存
        promoteToL1(key, entry);
        
        // 异步存储到L3缓存
        l3Cache.putAsync(key, value);
    }
    
    // 提升到L1缓存
    private void promoteToL1(String key, CacheEntry<T> entry) {
        l1Cache.put(key, entry);
        
        // 检查L1缓存大小限制
        if (l1Cache.size() > l1MaxSize) {
            evictFromL1();
        }
    }
    
    // 从L1缓存中淘汰条目
    private void evictFromL1() {
        // 找到价值最低的条目
        String evictKey = l1Cache.entrySet().stream()
                .min((e1, e2) -> Double.compare(
                    e1.getValue().getCacheValue(), 
                    e2.getValue().getCacheValue()))
                .map(Map.Entry::getKey)
                .orElse(null);
        
        if (evictKey != null) {
            CacheEntry<T> evicted = l1Cache.remove(evictKey);
            
            // 降级到L2缓存
            if (evicted != null) {
                demoteToL2(evictKey, evicted);
            }
        }
    }
    
    // 降级到L2缓存
    private void demoteToL2(String key, CacheEntry<T> entry) {
        l2Cache.put(key, new SoftReference<>(entry));
        
        // 检查L2缓存大小限制
        if (l2Cache.size() > l2MaxSize) {
            evictFromL2();
        }
    }
    
    // 从L2缓存中淘汰条目
    private void evictFromL2() {
        // 简单的FIFO淘汰策略
        Iterator<String> iterator = l2Cache.keySet().iterator();
        if (iterator.hasNext()) {
            String key = iterator.next();
            l2Cache.remove(key);
        }
    }
    
    // 获取缓存统计信息
    public CacheStats getStats() {
        long totalRequests = l1Hits.get() + l2Hits.get() + l3Hits.get() + misses.get();
        
        return new CacheStats(
            l1Hits.get(), l2Hits.get(), l3Hits.get(), misses.get(),
            totalRequests, l1Cache.size(), l2Cache.size()
        );
    }
    
    // 清理缓存
    public void clear() {
        l1Cache.clear();
        l2Cache.clear();
        l3Cache.clear();
    }
    
    // 关闭缓存管理器
    public void shutdown() {
        l3Cache.shutdown();
    }
}

// 持久化缓存实现
class PersistentCache<T> {
    private final String cacheDir;
    private final ExecutorService asyncExecutor;
    
    public PersistentCache(String cacheDir) {
        this.cacheDir = cacheDir;
        this.asyncExecutor = Executors.newFixedThreadPool(2);
        
        // 确保缓存目录存在
        new File(cacheDir).mkdirs();
    }
    
    public T get(String key) {
        File cacheFile = new File(cacheDir, key + ".cache");
        if (!cacheFile.exists()) {
            return null;
        }
        
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(cacheFile))) {
            @SuppressWarnings("unchecked")
            T value = (T) ois.readObject();
            return value;
        } catch (IOException | ClassNotFoundException e) {
            // 缓存文件损坏，删除它
            cacheFile.delete();
            return null;
        }
    }
    
    public void put(String key, T value) {
        File cacheFile = new File(cacheDir, key + ".cache");
        
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(cacheFile))) {
            oos.writeObject(value);
        } catch (IOException e) {
            System.err.println("持久化缓存写入失败：" + e.getMessage());
        }
    }
    
    public void putAsync(String key, T value) {
        asyncExecutor.submit(() -> put(key, value));
    }
    
    public void clear() {
        File dir = new File(cacheDir);
        File[] files = dir.listFiles((file, name) -> name.endsWith(".cache"));
        
        if (files != null) {
            for (File file : files) {
                file.delete();
            }
        }
    }
    
    public void shutdown() {
        asyncExecutor.shutdown();
        try {
            if (!asyncExecutor.awaitTermination(30, TimeUnit.SECONDS)) {
                asyncExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            asyncExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}

// 缓存统计信息
class CacheStats {
    private final long l1Hits;
    private final long l2Hits;
    private final long l3Hits;
    private final long misses;
    private final long totalRequests;
    private final int l1Size;
    private final int l2Size;
    
    public CacheStats(long l1Hits, long l2Hits, long l3Hits, long misses, 
                     long totalRequests, int l1Size, int l2Size) {
        this.l1Hits = l1Hits;
        this.l2Hits = l2Hits;
        this.l3Hits = l3Hits;
        this.misses = misses;
        this.totalRequests = totalRequests;
        this.l1Size = l1Size;
        this.l2Size = l2Size;
    }
    
    public double getHitRate() {
        return totalRequests > 0 ? (double) (l1Hits + l2Hits + l3Hits) / totalRequests : 0.0;
    }
    
    public double getL1HitRate() {
        return totalRequests > 0 ? (double) l1Hits / totalRequests : 0.0;
    }
    
    @Override
    public String toString() {
        return String.format(
            "缓存统计 - 总命中率: %.2f%%, L1命中率: %.2f%%, L1大小: %d, L2大小: %d, 总请求: %d",
            getHitRate() * 100, getL1HitRate() * 100, l1Size, l2Size, totalRequests
        );
    }
}

// 代码分析结果（用于演示）
class CodeAnalysisResult implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private String codeHash;
    private String language;
    private String analysisResult;
    private LocalDateTime analyzedAt;
    
    public CodeAnalysisResult(String codeHash, String language, String analysisResult) {
        this.codeHash = codeHash;
        this.language = language;
        this.analysisResult = analysisResult;
        this.analyzedAt = LocalDateTime.now();
    }
    
    // getter 方法
    public String getCodeHash() { return codeHash; }
    public String getLanguage() { return language; }
    public String getAnalysisResult() { return analysisResult; }
    public LocalDateTime getAnalyzedAt() { return analyzedAt; }
    
    @Override
    public String toString() {
        return String.format("[%s] %s代码分析：%s", 
                           analyzedAt.format(DateTimeFormatter.ofPattern("MM-dd HH:mm")),
                           language, 
                           analysisResult.substring(0, Math.min(50, analysisResult.length())));
    }
}

// 代码分析缓存管理器
class CodeAnalysisCacheManager {
    private final MultiLevelCacheManager<CodeAnalysisResult> cacheManager;
    private final ChatModel model;
    
    public CodeAnalysisCacheManager(ChatModel model) {
        this.model = model;
        this.cacheManager = new MultiLevelCacheManager<>(50, 200, "./code_analysis_cache");
    }
    
    // 分析代码（带缓存）
    public CodeAnalysisResult analyzeCode(String code, String language) {
        // 生成代码哈希作为缓存键
        String codeHash = generateCodeHash(code);
        
        // 先从缓存中查找
        CodeAnalysisResult cached = cacheManager.get(codeHash);
        if (cached != null) {
            System.out.println("从缓存获取分析结果：" + codeHash);
            return cached;
        }
        
        // 缓存未命中，执行实际分析
        System.out.println("执行新的代码分析：" + codeHash);
        String analysisResult = performCodeAnalysis(code, language);
        
        // 创建分析结果并缓存
        CodeAnalysisResult result = new CodeAnalysisResult(codeHash, language, analysisResult);
        
        // 根据代码复杂度设置缓存优先级
        int priority = calculateCachePriority(code, language);
        cacheManager.put(codeHash, result, priority);
        
        return result;
    }
    
    // 执行实际的代码分析
    private String performCodeAnalysis(String code, String language) {
        try {
            String prompt = String.format(
                "请分析以下%s代码的质量、潜在问题和改进建议：\n```%s\n%s\n```",
                language, language, code
            );
            
            return model.chat(prompt);
        } catch (Exception e) {
            return "代码分析失败：" + e.getMessage();
        }
    }
    
    // 生成代码哈希
    private String generateCodeHash(String code) {
        // 简单的哈希实现，实际应用中应使用更好的哈希算法
        return "hash_" + Math.abs(code.hashCode());
    }
    
    // 计算缓存优先级
    private int calculateCachePriority(String code, String language) {
        int priority = 1;
        
        // 根据代码长度调整优先级
        if (code.length() > 1000) priority += 2;
        else if (code.length() > 500) priority += 1;
        
        // 根据语言类型调整优先级
        if ("java".equalsIgnoreCase(language) || "python".equalsIgnoreCase(language)) {
            priority += 1;
        }
        
        // 根据代码复杂度调整优先级
        if (code.contains("class") || code.contains("function") || code.contains("def")) {
            priority += 1;
        }
        
        return Math.min(priority, 5); // 最大优先级为5
    }
    
    // 获取缓存统计
    public CacheStats getCacheStats() {
        return cacheManager.getStats();
    }
    
    // 清理缓存
    public void clearCache() {
        cacheManager.clear();
    }
    
    // 关闭管理器
    public void shutdown() {
        cacheManager.shutdown();
    }
}

public class CacheAndPersistenceExample {
    public static void main(String[] args) {
        ChatModel model = QwenChatModel.builder()
                .apiKey("your-api-key")
                .modelName("qwen-max")
                .temperature(0.4f)
                .build();
        
        CodeAnalysisCacheManager cacheManager = new CodeAnalysisCacheManager(model);
        
        System.out.println("=== 代码小抄缓存系统演示 ===");
        
        // 测试代码样本
        String[] testCodes = {
                "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\"); } }",
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "function quickSort(arr) { if (arr.length <= 1) return arr; /* 快排实现 */ }",
                "class Calculator { constructor() { this.result = 0; } add(x) { this.result += x; return this; } }",
                "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\"); } }" // 重复代码，测试缓存
        };
        
        String[] languages = {"java", "python", "javascript", "javascript", "java"};
        
        // 第一轮分析（全部缓存未命中）
        System.out.println("--- 第一轮分析 ---");
        for (int i = 0; i < testCodes.length; i++) {
            System.out.println("\n分析代码 " + (i + 1) + ":");
            CodeAnalysisResult result = cacheManager.analyzeCode(testCodes[i], languages[i]);
            System.out.println("结果：" + result);
        }
        
        System.out.println("\n" + cacheManager.getCacheStats());
        
        // 第二轮分析（部分缓存命中）
        System.out.println("\n--- 第二轮分析（测试缓存效果）---");
        for (int i = 0; i < 3; i++) {
            System.out.println("\n重新分析代码 " + (i + 1) + ":");
            CodeAnalysisResult result = cacheManager.analyzeCode(testCodes[i], languages[i]);
            System.out.println("结果：" + result);
        }
        
        System.out.println("\n" + cacheManager.getCacheStats());
        
        // 添加更多代码以测试缓存淘汰
        System.out.println("\n--- 测试缓存淘汰机制 ---");
        for (int i = 0; i < 60; i++) {
            String code = "// 测试代码 " + i + "\npublic void test" + i + "() { System.out.println(\"Test " + i + "\"); }";
            cacheManager.analyzeCode(code, "java");
            
            if (i % 20 == 19) {
                System.out.println("添加了 " + (i + 1) + " 个新分析结果");
                System.out.println(cacheManager.getCacheStats());
            }
        }
        
        System.out.println("\n=== 最终缓存统计 ===");
        System.out.println(cacheManager.getCacheStats());
        
        cacheManager.shutdown();
    }
}
