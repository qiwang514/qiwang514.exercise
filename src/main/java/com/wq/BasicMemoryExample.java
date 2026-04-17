package com.wq;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.community.model.dashscope.QwenChatModel;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

// 基础记忆接口
interface BasicMemory {
    void addMessage(ChatMessage message);
    List<ChatMessage> getMessages();
    void clear();
    int getMessageCount();
    boolean isEmpty();
}

// 简单记忆实现
class SimpleMemoryImpl implements BasicMemory {
    private final List<ChatMessage> messages;
    private final int maxCapacity;
    
    public SimpleMemoryImpl(int maxCapacity) {
        this.messages = new ArrayList<>();
        this.maxCapacity = maxCapacity;
    }
    
    @Override
    public synchronized void addMessage(ChatMessage message) {
        messages.add(message);
        
        // 当超过容量限制时，移除最早的消息
        while (messages.size() > maxCapacity) {
            messages.remove(0);
        }
    }
    
    @Override
    public synchronized List<ChatMessage> getMessages() {
        return new ArrayList<>(messages);
    }
    
    @Override
    public synchronized void clear() {
        messages.clear();
    }
    
    @Override
    public synchronized int getMessageCount() {
        return messages.size();
    }
    
    @Override
    public synchronized boolean isEmpty() {
        return messages.isEmpty();
    }
}


 
// 带时间戳的记忆条目
class TimestampedMessage {
    private final ChatMessage message;
    private final LocalDateTime timestamp;
    private final String messageId;

    public TimestampedMessage(ChatMessage message) {
        this.message = message;
        this.timestamp = LocalDateTime.now();
        this.messageId = "msg_" + System.currentTimeMillis() + "_" + Math.random();
    }

    // getter 方法
    public ChatMessage getMessage() {
        return message;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public String getMessageId() {
        return messageId;
    }

    @Override
    public String toString() {
        return String.format("[%s] %s: %s",
                timestamp.toString(),
                message.type(),
                getMessageContent(message).substring(0, Math.min(50, getMessageContent(message).length())));
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


// 面试记忆管理器
class InterviewMemoryManager {
    private final Map<String, BasicMemory> candidateMemories;
    private final ChatModel model;

    public InterviewMemoryManager(ChatModel model) {
        this.candidateMemories = new ConcurrentHashMap<>();
        this.model = model;
    }

    // 为候选人创建或获取记忆
    public BasicMemory getOrCreateMemory(String candidateId) {
        return candidateMemories.computeIfAbsent(candidateId,
                k -> new SimpleMemoryImpl(20)); // 每个候选人最多记住20条消息
    }

    // 开始面试
    public String startInterview(String candidateId, String position) {
        BasicMemory memory = getOrCreateMemory(candidateId);

        // 添加系统消息设定面试场景
        SystemMessage systemMsg = new SystemMessage(
                "你是面试鸭的专业面试官，正在面试" + position + "职位的候选人。" +
                        "请保持专业、友好的态度，根据对话历史提出有针对性的问题。"
        );
        memory.addMessage(systemMsg);

        String welcomeMessage = "欢迎参加" + position + "职位的面试！" +
                "我是您今天的面试官，让我们开始吧。请先简单介绍一下自己。";

        // 记录面试官的欢迎消息
        AiMessage welcomeMsg = new AiMessage(welcomeMessage);
        memory.addMessage(welcomeMsg);

        System.out.println("为候选人 " + candidateId + " 创建面试记忆，当前消息数：" + memory.getMessageCount());

        return welcomeMessage;
    }

    // 处理候选人回答
    public String processCandidateAnswer(String candidateId, String answer) {
        BasicMemory memory = getOrCreateMemory(candidateId);

        // 记录候选人的回答
        UserMessage userMsg = new UserMessage(answer);
        memory.addMessage(userMsg);

        // 基于历史对话生成面试官回应
        String context = buildContextFromMemory(memory);
        String response = generateInterviewerResponse(context, answer);

        // 记录面试官的回应
        AiMessage aiMsg = new AiMessage(response);
        memory.addMessage(aiMsg);

        System.out.println("候选人 " + candidateId + " 回答已记录，当前消息数：" + memory.getMessageCount());

        return response;
    }

    // 从记忆中构建上下文
    private String buildContextFromMemory(BasicMemory memory) {
        List<ChatMessage> messages = memory.getMessages();
        StringBuilder context = new StringBuilder();

        context.append("对话历史：\n");
        for (ChatMessage msg : messages) {
            if (msg instanceof SystemMessage) {
                continue; // 跳过系统消息
            }

            String role = msg instanceof UserMessage ? "候选人" : "面试官";
            String content = getMessageContent(msg);
            context.append(role).append("：").append(content).append("\n");
        }

        return context.toString();
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

    // 生成面试官回应
    private String generateInterviewerResponse(String context, String currentAnswer) {
        try {
            String prompt = context + "\n候选人刚才的回答：" + currentAnswer +
                    "\n\n请作为面试官，基于对话历史给出专业的回应或提出下一个问题。";

            return model.chat(prompt);
        } catch (Exception e) {
            return "感谢您的回答。请继续介绍您的技术背景。";
        }
    }

    // 获取面试摘要
    public String getInterviewSummary(String candidateId) {
        BasicMemory memory = candidateMemories.get(candidateId);
        if (memory == null || memory.isEmpty()) {
            return "该候选人尚未开始面试。";
        }

        String context = buildContextFromMemory(memory);
        String prompt = "请基于以下面试对话，生成简洁的面试摘要：\n" + context;

        try {
            return model.chat(prompt);
        } catch (Exception e) {
            return "面试摘要生成失败。";
        }
    }

    // 清除候选人记忆
    public void clearCandidateMemory(String candidateId) {
        BasicMemory memory = candidateMemories.get(candidateId);
        if (memory != null) {
            memory.clear();
            System.out.println("已清除候选人 " + candidateId + " 的面试记忆");
        }
    }

    // 获取统计信息
    public Map<String, Integer> getMemoryStats() {
        Map<String, Integer> stats = new ConcurrentHashMap<>();

        candidateMemories.forEach((candidateId, memory) -> {
            stats.put(candidateId, memory.getMessageCount());
        });

        return stats;
    }
}

public class BasicMemoryExample {
    public static void main(String[] args) {
        // 初始化通义千问模型
        ChatModel model = QwenChatModel.builder()
                .apiKey("your-api-key")
                .modelName("qwen-max")
                .temperature(0.7f)
                .build();
        
        // 创建面试记忆管理器
        InterviewMemoryManager memoryManager = new InterviewMemoryManager(model);
        
        String candidateId = "candidate_yupi";
        String position = "Java 后端开发工程师";
        
        System.out.println("=== 面试鸭记忆功能演示 ===");
        
        // 开始面试
        System.out.println("--- 开始面试 ---");
        String welcomeMessage = memoryManager.startInterview(candidateId, position);
        System.out.println("面试官：" + welcomeMessage);
        
        // 模拟多轮对话
        String[] candidateAnswers = {
                "我是程序员鱼皮，有5年Java开发经验，主要做过编程导航和面试鸭等项目。",
                "我熟悉Spring Boot、MySQL、Redis等技术栈，有丰富的Web开发经验。",
                "在编程导航项目中，我负责后端架构设计和核心功能开发，日活用户达到10万+。",
                "我认为我的优势是学习能力强，能够快速掌握新技术，并且有良好的代码规范。"
        };
        
        for (int i = 0; i < candidateAnswers.length; i++) {
            System.out.println("\n--- 第 " + (i + 1) + " 轮对话 ---");
            System.out.println("候选人：" + candidateAnswers[i]);
            
            String response = memoryManager.processCandidateAnswer(candidateId, candidateAnswers[i]);
            System.out.println("面试官：" + response);
        }
        
        // 显示记忆统计
        System.out.println("\n--- 记忆统计 ---");
        Map<String, Integer> stats = memoryManager.getMemoryStats();
        stats.forEach((id, count) -> 
            System.out.println("候选人 " + id + "：" + count + " 条消息"));
        
        // 生成面试摘要
        System.out.println("\n--- 面试摘要 ---");
        String summary = memoryManager.getInterviewSummary(candidateId);
        System.out.println(summary);
        
        // 清除记忆
        memoryManager.clearCandidateMemory(candidateId);
    }
}
