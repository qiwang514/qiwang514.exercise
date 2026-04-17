package com.wq;

import dev.langchain4j.community.model.dashscope.QwenChatModel;
import dev.langchain4j.model.chat.ChatLanguageModel;

public class LangChainAiInvoke {

    public static void main(String[] args) {
        ChatLanguageModel qwenModel = QwenChatModel.builder()
                .apiKey(TestApiKey.API_KEY)
                .modelName("qwen-max")
                .build();
        String answer = qwenModel.chat("你是谁");
        String answer1 = qwenModel.chat("我是王琦");
        String answer2 = qwenModel.chat("你知道我是谁吗");
        System.out.println(answer);
        System.out.println(answer1);
        System.out.println(answer2);
    }
}
