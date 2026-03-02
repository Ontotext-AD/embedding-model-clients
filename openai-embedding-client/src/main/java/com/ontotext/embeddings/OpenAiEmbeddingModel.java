package com.ontotext.embeddings;

import com.ontotext.Config;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class OpenAiEmbeddingModel implements EmbeddingModel {

    public static final String MODEL_NAME_PROPERTY = "openai.embedding.model.name";
    public static final String API_KEY_PROPERTY = "openai.embedding.model.api.key";
    public static final String DIMENSIONS_PROPERTY = "openai.embedding.model.dimensions";
    public static final String URI_PROPERTY = "openai.embedding.model.baseUrl";
    private static final Logger LOGGER = LoggerFactory.getLogger(OpenAiEmbeddingModel.class);

    private final EmbeddingModel embeddingModel;
    private final String modelName;

    public OpenAiEmbeddingModel(String modelName) {
        this.modelName = resolveModelName(modelName);
        this.embeddingModel = createEmbeddingModel();
    }

    //Used in GraphDB 11.2.x and 11.3.x
    @Deprecated
    public OpenAiEmbeddingModel() {
        this(Config.getProperty(MODEL_NAME_PROPERTY));
    }
    private String resolveModelName(String modelName) {
        if (modelName == null || modelName.isEmpty()) {
           modelName = Config.getProperty(MODEL_NAME_PROPERTY);
        }
        return modelName;
    }

    private EmbeddingModel createEmbeddingModel() {
        String uri = null; // The builder will throw an error if null.
        if (Config.getProperty(URI_PROPERTY) != null) {
            // Strip trailing slash.
            uri = Config.getProperty(URI_PROPERTY).replaceAll("/+$", "");
        }
        dev.langchain4j.model.openai.OpenAiEmbeddingModel.OpenAiEmbeddingModelBuilder builder =
                dev.langchain4j.model.openai.OpenAiEmbeddingModel.builder().baseUrl(uri).modelName(modelName)
                        .apiKey(Config.getProperty(API_KEY_PROPERTY));
        String dimensions = Config.getProperty(DIMENSIONS_PROPERTY);
        if (dimensions != null && !dimensions.isEmpty()) {
            builder.dimensions(Config.getPropertyInt(DIMENSIONS_PROPERTY));
        }
        LOGGER.info("Creating OpenAI Embedding model with endpoint {} and deployment {}.", uri, modelName);
        if (LOGGER.isDebugEnabled()) {
            LOGGER.info("Will log requests and responses.");
            builder.logRequests(true);
            builder.logResponses(true);
        }
        return builder.build();
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> list) {
        return embeddingModel.embedAll(list);
    }
}
