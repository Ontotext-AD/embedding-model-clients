package com.ontotext.embeddings;

import com.ontotext.Config;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.azure.AzureOpenAiEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AzureEmbeddingModel implements EmbeddingModel {

  public static final String MODEL_NAME = "openai.embedding.model.name";
  public static final String API_KEY = "openai.embedding.model.api.key";
  public static final String DIMENSIONS = "openai.embedding.model.dimensions";
  public static final String URI = "openai.embedding.model.baseUrl";
  private static final Logger LOGGER = LoggerFactory.getLogger(AzureEmbeddingModel.class);
  private final EmbeddingModel embeddingModel;

  public AzureEmbeddingModel() {
    this.embeddingModel = createEmbeddingModel();
  }


  private static EmbeddingModel createEmbeddingModel() {
    String uri = null; // Let it throw the Azure error if null.
    if (Config.getProperty(URI) != null) {
      // Strip trailing slash.
      uri = Config.getProperty(URI).replaceAll("/+$", "");
    }
    String deployment = Config.getProperty(MODEL_NAME);
    AzureOpenAiEmbeddingModel.Builder builder =
        AzureOpenAiEmbeddingModel.builder().deploymentName(deployment).endpoint(uri)
            .apiKey(Config.getProperty(API_KEY));
    String dimensions = Config.getProperty(DIMENSIONS);
    LOGGER.info("Creating Azure Embedding model with endpoint {} and deployment {}.", uri,
        deployment);
    if (LOGGER.isDebugEnabled()) {
      LOGGER.info("Will log requests and responses.");
      builder.logRequestsAndResponses(true);
    }
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
