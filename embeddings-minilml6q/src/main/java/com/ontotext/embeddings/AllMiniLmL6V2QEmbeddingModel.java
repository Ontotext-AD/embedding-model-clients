package com.ontotext.embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.output.Response;

import java.util.List;

public class AllMiniLmL6V2QEmbeddingModel implements EmbeddingModel {

    private AllMiniLmL6V2QuantizedEmbeddingModel embeddingModel;

    public AllMiniLmL6V2QEmbeddingModel() {
        this.embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        return embeddingModel.embedAll(textSegments);
    }

    @Override
    public Response<Embedding> embed(String text) {
        return embeddingModel.embed(text);
    }

    @Override
    public Response<Embedding> embed(TextSegment textSegment) {
        return embeddingModel.embed(textSegment);
    }

    @Override
    public int dimension() {
        return embeddingModel.dimension();
    }
}
