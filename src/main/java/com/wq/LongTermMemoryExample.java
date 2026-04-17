package com.wq;

import dev.langchain4j.community.model.dashscope.QwenChatModel;
import dev.langchain4j.model.chat.ChatModel;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

// 长期记忆条目
class LongTermMemoryEntry implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private String entryId;
    private String userId;
    private String category;
    private String content;
    private LocalDateTime createdAt;
    private LocalDateTime lastAccessedAt;
    private int accessCount;
    private double importance;
    
    public LongTermMemoryEntry(String userId, String category, String content, double importance) {
        this.entryId = "entry_" + System.currentTimeMillis() + "_" + Math.random();
        this.userId = userId;
        this.category = category;
        this.content = content;
        this.importance = importance;
        this.createdAt = LocalDateTime.now();
        this.lastAccessedAt = LocalDateTime.now();
        this.accessCount = 1;
    }
    
    // 更新访问信息
    public void updateAccess() {
        this.lastAccessedAt = LocalDateTime.now();
        this.accessCount++;
    }
    
    // 计算记忆价值（基于重要性、访问频率和时间衰减）
    public double calculateMemoryValue() {
        long daysSinceCreated = java.time.Duration.between(createdAt, LocalDateTime.now()).toDays();
        long daysSinceAccessed = java.time.Duration.between(lastAccessedAt, LocalDateTime.now()).toDays();
        
        // 时间衰减因子
        double timeDecay = Math.exp(-daysSinceAccessed * 0.1);
        
        // 访问频率加权
        double accessWeight = Math.log(accessCount + 1);
        
        return importance * timeDecay * accessWeight;
    }
    
    // getter 和 setter 方法
    public String getEntryId() { return entryId; }
    public String getUserId() { return userId; }
    public String getCategory() { return category; }
    public String getContent() { return content; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getLastAccessedAt() { return lastAccessedAt; }
    public int getAccessCount() { return accessCount; }
    public double getImportance() { return importance; }
    public void setImportance(double importance) { this.importance = importance; }
    
    @Override
    public String toString() {
        return String.format("[%s] %s: %s (重要性: %.2f, 访问: %d次)", 
                           category, userId, content.substring(0, Math.min(50, content.length())), 
                           importance, accessCount);
    }
}

// 长期记忆管理器
class LongTermMemoryManager {
    private final Map<String, List<LongTermMemoryEntry>> userMemories;
    private final Map<String, LongTermMemoryEntry> entryIndex;
    private final ScheduledExecutorService scheduler;
    private final String persistenceFile;
    private final int maxEntriesPerUser;
    private final ChatModel model;
    
    public LongTermMemoryManager(ChatModel model, String persistenceFile, int maxEntriesPerUser) {
        this.model = model;
        this.persistenceFile = persistenceFile;
        this.maxEntriesPerUser = maxEntriesPerUser;
        this.userMemories = new ConcurrentHashMap<>();
        this.entryIndex = new ConcurrentHashMap<>();
        this.scheduler = Executors.newScheduledThreadPool(2);
        
        // 启动定期清理和持久化任务
        scheduler.scheduleAtFixedRate(this::performMaintenance, 1, 1, TimeUnit.HOURS);
        scheduler.scheduleAtFixedRate(this::persistMemories, 10, 10, TimeUnit.MINUTES);
        
        // 启动时加载历史记忆
        loadMemories();
    }
    
    // 添加长期记忆
    public void addLongTermMemory(String userId, String category, String content, double importance) {
        LongTermMemoryEntry entry = new LongTermMemoryEntry(userId, category, content, importance);
        
        userMemories.computeIfAbsent(userId, k -> new ArrayList<>()).add(entry);
        entryIndex.put(entry.getEntryId(), entry);
        
        // 检查用户记忆条目数量限制
        List<LongTermMemoryEntry> userEntries = userMemories.get(userId);
        if (userEntries.size() > maxEntriesPerUser) {
            // 移除价值最低的记忆条目
            LongTermMemoryEntry leastValuable = userEntries.stream()
                    .min((a, b) -> Double.compare(a.calculateMemoryValue(), b.calculateMemoryValue()))
                    .orElse(null);
            
            if (leastValuable != null) {
                userEntries.remove(leastValuable);
                entryIndex.remove(leastValuable.getEntryId());
                System.out.println("移除低价值记忆：" + leastValuable.getContent().substring(0, Math.min(30, leastValuable.getContent().length())));
            }
        }
        
        System.out.println("添加长期记忆：" + category + " - " + content.substring(0, Math.min(50, content.length())));
    }
    
    // 检索相关记忆
    public List<LongTermMemoryEntry> retrieveRelevantMemories(String userId, String query, int maxResults) {
        List<LongTermMemoryEntry> userEntries = userMemories.getOrDefault(userId, new ArrayList<>());
        
        // 使用AI模型进行语义相似度匹配
        List<ScoredMemory> scoredMemories = new ArrayList<>();
        
        for (LongTermMemoryEntry entry : userEntries) {
            double relevanceScore = calculateRelevanceScore(query, entry.getContent());
            double memoryValue = entry.calculateMemoryValue();
            double combinedScore = relevanceScore * 0.7 + memoryValue * 0.3;
            
            scoredMemories.add(new ScoredMemory(entry, combinedScore));
            entry.updateAccess(); // 更新访问信息
        }
        
        // 按综合得分排序并返回前N个
        return scoredMemories.stream()
                .sorted((a, b) -> Double.compare(b.score, a.score))
                .limit(maxResults)
                .map(sm -> sm.entry)
                .collect(Collectors.toList());
    }
    
    // 计算相关性得分
    private double calculateRelevanceScore(String query, String content) {
        try {
            String prompt = String.format(
                "请评估以下查询与内容的相关性，返回0-1之间的数值：\n查询：%s\n内容：%s\n相关性得分：",
                query, content
            );
            
            String response = model.chat(prompt);
            
            // 尝试从响应中提取数值
            Pattern pattern = Pattern.compile("([0-9]*\\.?[0-9]+)");
            Matcher matcher = pattern.matcher(response);
            if (matcher.find()) {
                double score = Double.parseDouble(matcher.group(1));
                return Math.min(1.0, Math.max(0.0, score));
            }
        } catch (Exception e) {
            // 如果AI评估失败，使用简单的关键词匹配
            return simpleKeywordMatch(query, content);
        }
        
        return 0.0;
    }
    
    // 简单关键词匹配
    private double simpleKeywordMatch(String query, String content) {
        String[] queryWords = query.toLowerCase().split("\\s+");
        String lowerContent = content.toLowerCase();
        
        int matches = 0;
        for (String word : queryWords) {
            if (lowerContent.contains(word)) {
                matches++;
            }
        }
        
        return (double) matches / queryWords.length;
    }
    
    // 生成基于长期记忆的学习建议
    public String generateLearningAdvice(String userId, String currentTopic) {
        List<LongTermMemoryEntry> relevantMemories = retrieveRelevantMemories(userId, currentTopic, 5);
        
        StringBuilder context = new StringBuilder();
        context.append("用户学习历史：\n");
        
        for (LongTermMemoryEntry memory : relevantMemories) {
            context.append("- [").append(memory.getCategory()).append("] ")
                   .append(memory.getContent()).append("\n");
        }
        
        context.append("\n当前学习主题：").append(currentTopic);
        context.append("\n\n请基于用户的学习历史，为当前主题提供个性化的学习建议。");
        
        try {
            return model.chat(context.toString());
        } catch (Exception e) {
            return "基于您的学习历史，建议循序渐进地学习" + currentTopic + "相关知识。";
        }
    }
    
    // 定期维护
    private void performMaintenance() {
        System.out.println("开始长期记忆维护...");
        
        int totalEntries = entryIndex.size();
        int removedEntries = 0;
        
        // 清理过期或低价值的记忆条目
        List<String> toRemove = new ArrayList<>();
        
        for (LongTermMemoryEntry entry : entryIndex.values()) {
            // 移除超过一年未访问且价值很低的条目
            long daysSinceAccessed = java.time.Duration.between(entry.getLastAccessedAt(), LocalDateTime.now()).toDays();
            
            if (daysSinceAccessed > 365 && entry.calculateMemoryValue() < 0.1) {
                toRemove.add(entry.getEntryId());
            }
        }
        
        for (String entryId : toRemove) {
            LongTermMemoryEntry entry = entryIndex.remove(entryId);
            if (entry != null) {
                List<LongTermMemoryEntry> userEntries = userMemories.get(entry.getUserId());
                if (userEntries != null) {
                    userEntries.remove(entry);
                }
                removedEntries++;
            }
        }
        
        System.out.println("维护完成：总条目 " + totalEntries + "，清理 " + removedEntries + " 条");
    }
    
    // 持久化记忆
    private void persistMemories() {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(persistenceFile))) {
            Map<String, Object> data = new HashMap<>();
            data.put("userMemories", userMemories);
            data.put("entryIndex", entryIndex);
            data.put("timestamp", LocalDateTime.now());
            
            oos.writeObject(data);
            System.out.println("记忆已持久化到文件：" + persistenceFile);
        } catch (IOException e) {
            System.err.println("持久化失败：" + e.getMessage());
        }
    }
    
    // 加载记忆
    @SuppressWarnings("unchecked")
    private void loadMemories() {
        File file = new File(persistenceFile);
        if (!file.exists()) {
            System.out.println("未找到历史记忆文件，从空白状态开始");
            return;
        }
        
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(persistenceFile))) {
            Map<String, Object> data = (Map<String, Object>) ois.readObject();
            
            Map<String, List<LongTermMemoryEntry>> loadedUserMemories = 
                (Map<String, List<LongTermMemoryEntry>>) data.get("userMemories");
            Map<String, LongTermMemoryEntry> loadedEntryIndex = 
                (Map<String, LongTermMemoryEntry>) data.get("entryIndex");
            
            if (loadedUserMemories != null) {
                userMemories.putAll(loadedUserMemories);
            }
            if (loadedEntryIndex != null) {
                entryIndex.putAll(loadedEntryIndex);
            }
            
            LocalDateTime savedTime = (LocalDateTime) data.get("timestamp");
            System.out.println("成功加载长期记忆，保存时间：" + savedTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
            System.out.println("加载记忆条目：" + entryIndex.size() + " 条");
            
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("加载记忆失败：" + e.getMessage());
        }
    }
    
    // 获取用户记忆统计
    public Map<String, Object> getUserMemoryStats(String userId) {
        List<LongTermMemoryEntry> userEntries = userMemories.getOrDefault(userId, new ArrayList<>());
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalEntries", userEntries.size());
        stats.put("categories", userEntries.stream()
                .collect(Collectors.groupingBy(LongTermMemoryEntry::getCategory, Collectors.counting())));
        
        OptionalDouble avgImportance = userEntries.stream()
                .mapToDouble(LongTermMemoryEntry::getImportance)
                .average();
        stats.put("averageImportance", avgImportance.orElse(0.0));
        
        return stats;
    }
    
    // 关闭管理器
    public void shutdown() {
        scheduler.shutdown();
        persistMemories(); // 最后一次持久化
        
        try {
            if (!scheduler.awaitTermination(30, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    // 评分记忆内部类
    private static class ScoredMemory {
        final LongTermMemoryEntry entry;
        final double score;
        
        ScoredMemory(LongTermMemoryEntry entry, double score) {
            this.entry = entry;
            this.score = score;
        }
    }
}

public class LongTermMemoryExample {
    public static void main(String[] args) {
        ChatModel model = QwenChatModel.builder()
                .apiKey("your-api-key")
                .modelName("qwen-max")
                .temperature(0.5f)
                .build();
        
        LongTermMemoryManager memoryManager = new LongTermMemoryManager(
            model, "algorithm_learning_memory.dat", 100);
        
        String userId = "user_yupi";
        
        System.out.println("=== 算法导航长期记忆演示 ===");
        
        // 模拟用户学习历史的积累
        String[][] learningHistory = {
                {"学习进度", "完成了数组基础知识学习，掌握了基本操作", "0.8"},
                {"解题记录", "成功解决了两数之和问题，使用哈希表优化", "0.9"},
                {"知识点", "理解了时间复杂度和空间复杂度的概念", "0.7"},
                {"项目经验", "在编程导航项目中使用了ArrayList和HashMap", "0.8"},
                {"学习偏好", "喜欢通过实际项目来学习算法知识", "0.6"},
                {"困难点", "对递归算法的理解还不够深入", "0.9"},
                {"成就", "在面试鸭上完成了100道算法题", "0.8"},
                {"目标", "希望能够解决中等难度的算法题", "0.7"},
                {"学习方式", "偏好视频教程配合代码实践", "0.5"},
                {"技能树", "已掌握：数组、链表；正在学习：树和图", "0.9"}
        };
        
        // 添加学习记忆
        System.out.println("--- 建立学习记忆 ---");
        for (String[] record : learningHistory) {
            String category = record[0];
            String content = record[1];
            double importance = Double.parseDouble(record[2]);
            
            memoryManager.addLongTermMemory(userId, category, content, importance);
        }
        
        // 显示用户记忆统计
        System.out.println("\n--- 记忆统计 ---");
        Map<String, Object> stats = memoryManager.getUserMemoryStats(userId);
        System.out.println("总记忆条目：" + stats.get("totalEntries"));
        System.out.println("平均重要性：" + String.format("%.2f", (Double) stats.get("averageImportance")));
        
        @SuppressWarnings("unchecked")
        Map<String, Long> categories = (Map<String, Long>) stats.get("categories");
        System.out.println("分类统计：");
        categories.forEach((category, count) -> 
            System.out.println("  " + category + ": " + count + " 条"));
        
        // 测试记忆检索
        System.out.println("\n--- 记忆检索测试 ---");
        String[] queries = {
                "二叉树算法",
                "项目实战经验",
                "递归相关问题"
        };
        
        for (String query : queries) {
            System.out.println("\n查询：" + query);
            List<LongTermMemoryEntry> relevantMemories = 
                memoryManager.retrieveRelevantMemories(userId, query, 3);
            
            for (int i = 0; i < relevantMemories.size(); i++) {
                LongTermMemoryEntry memory = relevantMemories.get(i);
                System.out.println("  " + (i + 1) + ". " + memory);
            }
        }
        
        // 生成学习建议
        System.out.println("\n--- 个性化学习建议 ---");
        String currentTopic = "二叉树遍历算法";
        String advice = memoryManager.generateLearningAdvice(userId, currentTopic);
        System.out.println("当前学习主题：" + currentTopic);
        System.out.println("AI建议：" + advice);
        
        // 关闭管理器
        memoryManager.shutdown();
    }
}
