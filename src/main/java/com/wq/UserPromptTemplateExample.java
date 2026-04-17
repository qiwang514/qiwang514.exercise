package com.wq;



import dev.langchain4j.community.model.dashscope.QwenChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

interface ResumeAssistant {
    @SystemMessage("你是老鱼简历网站的简历优化专家，擅长针对{industry}行业优化简历。")
    @UserMessage("请帮我优化以下{position}岗位的简历部分:\n\n{resumeContent}")
    String optimizeResume(@V("resumeContent") String resumeContent,
                          @V("industry") String industry,
                          @V("position") String position);
}

public class UserPromptTemplateExample {
    public static void main(String[] args) {
        // 初始化通义千问模型
        ChatModel model = QwenChatModel.builder()
                .apiKey(TestApiKey.API_KEY) // 替换为你自己的 DashScope API Key
                .modelName("qwen-max")
                .temperature(0.7f)
                .build();

        // 创建服务
        ResumeAssistant assistant = AiServices.builder(ResumeAssistant.class)
                .chatModel(model)
                .build();

        // 示例简历内容
        String resumeContent = "工作经验:\n2020-2023 某科技公司 软件工程师\n负责开发和维护公司核心产品，使用Java和Spring框架。";

        String optimized = assistant.optimizeResume(resumeContent, "互联网", "后端开发");

        System.out.println("优化后的简历:\n" + optimized);
        System.out.println("commit");
        System.out.println("第三次提交");
    }
}

