package com.wq;

import dev.langchain4j.community.model.dashscope.QwenChatModel;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.response.ChatResponse;

import java.util.ArrayList;
import java.util.List;

public class MultiTurnChatExample {

    public static void main(String[] args) {
        // 初始化聊天模型
        ChatLanguageModel model = QwenChatModel.builder()
            .apiKey(TestApiKey.API_KEY)
            .modelName("qwen-max")
            .temperature(0.7f)
            .build();

        // 构建对话历史
        List<ChatMessage> messages = new ArrayList<>();
        // 添加系统消息，设定角色和行为边界
        messages.add(new SystemMessage("你是面试鸭的AI助手，专注于提供Java面试题解答"));
        // 添加用户问题
        messages.add(new UserMessage("你好，我是王琦。"));

        // 先拿 ChatResponse，再从里面拿 AiMessage
        ChatResponse chatResponse = model.chat(messages);
        AiMessage response = chatResponse.aiMessage();

        /**
        response.text()返回的是文本内容
        否则直接输出response的话 会返回AiMessage{text='Java 多线程是指...', ...} 是对象的 toString () 格式
        **/
        System.out.println("AI 回答: " + response.text());

        // 将AI回答添加到对话历史
        messages.add(response);
        System.out.println("AI回答添加到对话历史");
        // 继续对话，提出后续问题
        messages.add(new UserMessage("你知道我是谁吗 "));
        // 再次调用获取回答 AI会阅读整个list 并且只回答最后面的UserMessage
        ChatResponse chatResponse2 = model.chat(messages);
        AiMessage response2 = chatResponse2.aiMessage();
        System.out.println("AI 回答: " + response2.text());

    }
}
