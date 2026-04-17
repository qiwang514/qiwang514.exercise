package com.wq;

import dev.langchain4j.community.model.dashscope.QwenChatModel;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;


import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.locks.ReentrantReadWriteLock;


// 滑动窗口短期记忆实现
class SlidingWindowMemory implements BasicMemory {
    private final Queue<ChatMessage> messageQueue;
    private final int windowSize;
    private final ReentrantReadWriteLock lock;

    public SlidingWindowMemory(int windowSize) {
        this.windowSize = windowSize;
        this.messageQueue = new LinkedList<>();
        this.lock = new ReentrantReadWriteLock();
    }

    @Override
    public void addMessage(ChatMessage message) {
        lock.writeLock().lock();
        try {
            messageQueue.offer(message);

            // 维护窗口大小
            while (messageQueue.size() > windowSize) {
                ChatMessage removed = messageQueue.poll();
                System.out.println("移除旧消息：" + getMessageContent(removed).substring(0, Math.min(30, getMessageContent(removed).length())) + "...");
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    private String getMessageContent(ChatMessage message) {
        if (message instanceof UserMessage userMessage) {
            return userMessage.singleText();
        } else if (message instanceof AiMessage aiMessage) {
            return aiMessage.text();
        } else if (message instanceof SystemMessage systemMessage) {
            return systemMessage.text();
        } else {
            return message.toString();
        }
    }
    @Override
    public List<ChatMessage> getMessages() {
        lock.readLock().lock();
        try {
            return new ArrayList<>(messageQueue);
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public void clear() {
        lock.writeLock().lock();
        try {
            messageQueue.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public int getMessageCount() {
        lock.readLock().lock();
        try {
            return messageQueue.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public boolean isEmpty() {
        lock.readLock().lock();
        try {
            return messageQueue.isEmpty();
        } finally {
            lock.readLock().unlock();
        }
    }

    // 获取窗口使用率
    public double getWindowUtilization() {
        lock.readLock().lock();
        try {
            return (double) messageQueue.size() / windowSize;
        } finally {
            lock.readLock().unlock();
        }
    }
}

// 基于令牌数量的短期记忆
class TokenBasedShortMemory implements BasicMemory {
    private final List<ChatMessage> messages;
    private final int maxTokens;
    private final ReentrantReadWriteLock lock;
    private int currentTokenCount;

    public TokenBasedShortMemory(int maxTokens) {
        this.maxTokens = maxTokens;
        this.messages = new ArrayList<>();
        this.lock = new ReentrantReadWriteLock();
        this.currentTokenCount = 0;
    }

    @Override
    public void addMessage(ChatMessage message) {
        lock.writeLock().lock();
        try {
            int messageTokens = estimateTokenCount(getMessageContent(message));
            messages.add(message);
            currentTokenCount += messageTokens;

            // 当超过令牌限制时，移除最旧的消息
            while (currentTokenCount > maxTokens && !messages.isEmpty()) {
                ChatMessage removed = messages.remove(0);
                int removedTokens = estimateTokenCount(getMessageContent(removed));
                currentTokenCount -= removedTokens;
                System.out.println("因令牌限制移除消息，节省 " + removedTokens + " 个令牌");
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    private String getMessageContent(ChatMessage message) {
        if (message instanceof UserMessage userMessage) {
            return userMessage.singleText();
        } else if (message instanceof AiMessage aiMessage) {
            return aiMessage.text();
        } else if (message instanceof SystemMessage systemMessage) {
            return systemMessage.text();
        } else {
            return message.toString();
        }
    }
    @Override
    public List<ChatMessage> getMessages() {
        lock.readLock().lock();
        try {
            return new ArrayList<>(messages);
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public void clear() {
        lock.writeLock().lock();
        try {
            messages.clear();
            currentTokenCount = 0;
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public int getMessageCount() {
        lock.readLock().lock();
        try {
            return messages.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public boolean isEmpty() {
        lock.readLock().lock();
        try {
            return messages.isEmpty();
        } finally {
            lock.readLock().unlock();
        }
    }

    // 简单的令牌数量估算
    private int estimateTokenCount(String text) {
        // 粗略估算：中文字符按1.5个令牌计算，英文单词按1个令牌计算
        int chineseChars = 0;
        int englishWords = 0;

        for (char c : text.toCharArray()) {
            if (c >= 0x4e00 && c <= 0x9fff) {
                chineseChars++;
            }
        }

        englishWords = text.split("\\s+").length;

        return (int) (chineseChars * 1.5 + englishWords);
    }

    // 获取当前令牌使用情况
    public TokenUsageInfo getTokenUsage() {
        lock.readLock().lock();
        try {
            return new TokenUsageInfo(currentTokenCount, maxTokens);
        } finally {
            lock.readLock().unlock();
        }
    }

    // 令牌使用信息
    public static class TokenUsageInfo {
        private final int usedTokens;
        private final int maxTokens;

        public TokenUsageInfo(int usedTokens, int maxTokens) {
            this.usedTokens = usedTokens;
            this.maxTokens = maxTokens;
        }

        public int getUsedTokens() { return usedTokens; }
        public int getMaxTokens() { return maxTokens; }
        public double getUsageRate() { return (double) usedTokens / maxTokens; }
        public int getAvailableTokens() { return maxTokens - usedTokens; }

        @Override
        public String toString() {
            return String.format("令牌使用：%d/%d (%.1f%%)",
                    usedTokens, maxTokens, getUsageRate() * 100);
        }
    }
}

// 剪切助手的智能分类记忆管理器
class ClipboardMemoryManager {
    private final SlidingWindowMemory recentActions;
    private final TokenBasedShortMemory contentMemory;
    private final ChatModel model;

    public ClipboardMemoryManager(ChatModel model) {
        this.model = model;
        this.recentActions = new SlidingWindowMemory(10); // 记住最近10个操作
        this.contentMemory = new TokenBasedShortMemory(2000); // 最多2000个令牌的内容记忆
    }

    // 记录用户操作
    public void recordUserAction(String action, String content) {
        UserMessage actionMsg = new UserMessage("操作：" + action + "，内容：" + content);
        recentActions.addMessage(actionMsg);

        // 如果是重要内容，也加入内容记忆
        if (isImportantContent(content)) {
            contentMemory.addMessage(actionMsg);
        }

        System.out.println("记录用户操作：" + action);
        System.out.println("窗口使用率：" + String.format("%.1f%%", recentActions.getWindowUtilization() * 100));
        System.out.println("令牌使用：" + contentMemory.getTokenUsage());
    }

    // 获取个性化建议
    public String getPersonalizedSuggestion(String currentContent) {
        List<ChatMessage> recentMessages = recentActions.getMessages();
        List<ChatMessage> contentMessages = contentMemory.getMessages();

        StringBuilder context = new StringBuilder();
        context.append("用户最近的操作历史：\n");

        for (ChatMessage msg : recentMessages) {
            context.append("- ").append(getMessageContent(msg)).append("\n");
        }

        context.append("\n重要内容记忆：\n");
        for (ChatMessage msg : contentMessages) {
            context.append("- ").append(getMessageContent(msg)).append("\n");
        }

        context.append("\n当前剪切板内容：").append(currentContent);
        context.append("\n\n请基于用户的使用习惯，为当前内容提供智能分类建议。");

        try {
            return model.chat(context.toString());
        } catch (Exception e) {
            return "基于您的使用习惯，建议将此内容分类为常用类别。";
        }
    }


    // 判断是否为重要内容
    private boolean isImportantContent(String content) {
        // 简单的重要性判断逻辑
        return content.length() > 50 ||
                content.contains("密码") ||
                content.contains("账号") ||
                content.contains("代码") ||
                content.contains("http");
    }

    // 获取记忆统计信息
    public String getMemoryStats() {
        StringBuilder stats = new StringBuilder();
        stats.append("=== 剪切助手记忆统计 ===\n");
        stats.append("近期操作记忆：").append(recentActions.getMessageCount()).append(" 条\n");
        stats.append("窗口使用率：").append(String.format("%.1f%%", recentActions.getWindowUtilization() * 100)).append("\n");
        stats.append("内容记忆：").append(contentMemory.getMessageCount()).append(" 条\n");
        stats.append(contentMemory.getTokenUsage().toString()).append("\n");

        return stats.toString();
    }

    // 清理记忆
    public void clearMemory() {
        recentActions.clear();
        contentMemory.clear();
        System.out.println("已清理所有短期记忆");
    }

    private String getMessageContent(ChatMessage message) {
        if (message instanceof UserMessage userMessage) {
            return userMessage.singleText();
        } else if (message instanceof AiMessage aiMessage) {
            return aiMessage.text();
        } else if (message instanceof SystemMessage systemMessage) {
            return systemMessage.text();
        } else {
            return message.toString();
        }
    }
}


public class ShortTermMemoryExample {
    public static void main(String[] args) {
        ChatModel model = QwenChatModel.builder()
                .apiKey("your-api-key")
                .modelName("qwen-max")
                .temperature(0.6f)
                .build();
        
        ClipboardMemoryManager memoryManager = new ClipboardMemoryManager(model);
        
        System.out.println("=== 剪切助手短期记忆演示 ===");
        
        // 模拟用户的一系列操作
        String[][] userActions = {
                {"复制", "程序员鱼皮的编程导航网站：https://www.codefather.cn"},
                {"复制", "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello\"); } }"},
                {"复制", "用户名：yupi，密码：123456"},
                {"分类", "将代码片段分类到开发类别"},
                {"复制", "面试鸭题库：算法题、数据结构、系统设计"},
                {"搜索", "查找包含Java关键字的内容"},
                {"复制", "老鱼简历模板下载链接"},
                {"复制", "SELECT * FROM users WHERE status = 'active'"},
                {"分类", "将SQL语句分类到数据库类别"},
                {"复制", "今天的会议记录：讨论了项目进度和技术方案"},
                {"删除", "清理过期的临时文件"},
                {"复制", "算法导航学习路径：数组->链表->树->图"}
        };
        
        // 执行操作并观察记忆变化
        for (int i = 0; i < userActions.length; i++) {
            String action = userActions[i][0];
            String content = userActions[i][1];
            
            System.out.println("\n--- 操作 " + (i + 1) + " ---");
            memoryManager.recordUserAction(action, content);
            
            // 每隔几个操作显示统计信息
            if ((i + 1) % 4 == 0) {
                System.out.println("\n" + memoryManager.getMemoryStats());
            }
        }
        
        // 测试个性化建议
        System.out.println("\n--- 个性化建议测试 ---");
        String testContent = "import java.util.*; public class Solution { public int[] twoSum(int[] nums, int target) { /* 算法实现 */ } }";
        String suggestion = memoryManager.getPersonalizedSuggestion(testContent);
        System.out.println("当前内容：" + testContent);
        System.out.println("智能建议：" + suggestion);
        
        // 最终统计
        System.out.println("\n" + memoryManager.getMemoryStats());
    }
}
