package com.ontotext.embeddings;

import com.ontotext.Config;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;

import java.util.List;

public class OpenAiEmbeddingModel implements EmbeddingModel {

    public static final String MODEL_NAME = "openai.embedding.model.name";
    public static final String API_KEY = "openai.embedding.model.api.key";
    public static final String DIMENSIONS = "openai.embedding.model.dimensions";

    private final EmbeddingModel embeddingModel;

    public OpenAiEmbeddingModel() {
        this.embeddingModel = createEmbeddingModel();
    }

    private static EmbeddingModel createEmbeddingModel() {
        dev.langchain4j.model.openai.OpenAiEmbeddingModel.OpenAiEmbeddingModelBuilder builder = dev.langchain4j.model.openai.OpenAiEmbeddingModel.builder()
                .modelName(Config.getProperty(MODEL_NAME))
                .apiKey(Config.getProperty(API_KEY));
        String dimensions = Config.getProperty(DIMENSIONS);
        if (dimensions != null && !dimensions.isEmpty()) {
            builder.dimensions(Config.getPropertyInt(DIMENSIONS));
        }
        return builder.build();
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> list) {
        return embeddingModel.embedAll(list);
    }
}
