package com.ontotext.embeddings;

import com.ontotext.Config;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.bedrock.BedrockCohereEmbeddingModel;
import dev.langchain4j.model.bedrock.BedrockCohereEmbeddingModel.InputType;
import dev.langchain4j.model.bedrock.BedrockTitanEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import software.amazon.awssdk.regions.Region;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class AwsBedrockEmbeddingModel implements EmbeddingModel {

	public static final String MODEL_NAME_PROPERTY = "aws.bedrock.embedding.model.name";
	public static final String API_KEY_PROPERTY = "aws.bedrock.embedding.model.api.key";
	public static final String API_SECRET_PROPERTY = "aws.bedrock.embedding.model.api.secret";
	public static final String REGION_PROPERTY = "aws.bedrock.embedding.model.region";
	private static final Logger LOGGER = LoggerFactory.getLogger(AwsBedrockEmbeddingModel.class);
	
	private static final String COHERE_PREFIX = "cohere";
	private static final String TITAN_PREFIX = "amazon.titan";

	private final EmbeddingModel embeddingModel;
	private final String modelName;

	public AwsBedrockEmbeddingModel(String modelName) {
		this.modelName = resolveModelName(modelName);
		this.embeddingModel = createEmbeddingModel();
	}

	// Used in GraphDB 11.2.x and 11.3.x
	@Deprecated
	public AwsBedrockEmbeddingModel() {
		this(Config.getProperty(MODEL_NAME_PROPERTY));
	}

	private String resolveModelName(String modelName) {
		if (modelName == null || modelName.isEmpty()) {
			modelName = Config.getProperty(MODEL_NAME_PROPERTY);
		}
		LOGGER.info("Creating instance using model: {}", modelName);
		return modelName;
	}

	private EmbeddingModel createEmbeddingModel() {
		if (modelName.startsWith(COHERE_PREFIX)) {
			return createCohereEmbeddingModel();
		} else  if(modelName.startsWith(TITAN_PREFIX)){
			return createTitanEmbeddingModel();
		} else {
			throw new IllegalArgumentException("Unsupported AWS Bedrock embedding model - only cohere* and amazon.titan* models are supported");
		}
	}

	private EmbeddingModel createTitanEmbeddingModel() {
		LOGGER.info("Creating Aws Bedrock Titan Embedding model with region {} and model {}.", Config.getProperty(REGION_PROPERTY),
				modelName);

		return BedrockTitanEmbeddingModel.builder().model(modelName).region(Region.of(Config.getProperty(REGION_PROPERTY))).build();
	}

	private EmbeddingModel createCohereEmbeddingModel() {
		LOGGER.info("Creating Aws Bedrock Cohere Embedding model with region {} and model {}.", Config.getProperty(REGION_PROPERTY),
				modelName);

		return BedrockCohereEmbeddingModel.builder()
				.model(modelName)
				.inputType(InputType.SEARCH_DOCUMENT)
				.region(Region.of(Config.getProperty(REGION_PROPERTY))).build();
	}

	@Override
	public Response<List<Embedding>> embedAll(List<TextSegment> list) {
		return embeddingModel.embedAll(list);
	}
}
