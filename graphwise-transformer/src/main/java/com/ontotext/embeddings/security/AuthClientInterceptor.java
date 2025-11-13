package com.ontotext.embeddings.security;

import io.grpc.*;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.HexFormat;

public class AuthClientInterceptor implements ClientInterceptor {

  private static final String TIMESTAMP_HEADER = "x-timestamp";
  private static final String SIGNATURE_HEADER = "x-signature";
  private static final String ALGORITHM = "HmacSHA256";
  private final String secret;

  public AuthClientInterceptor(String secret) {
    this.secret = secret;
  }

  @Override
  public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
          MethodDescriptor<ReqT, RespT> method,
          CallOptions callOptions,
          Channel next) {

    return new ForwardingClientCall.SimpleForwardingClientCall<>(next.newCall(method, callOptions)) {
      @Override
      public void start(Listener<RespT> responseListener, Metadata headers) {
        String timestamp = String.valueOf(Instant.now().getEpochSecond());
        String signature = encode(secret, timestamp);

        headers.put(Metadata.Key.of(TIMESTAMP_HEADER, Metadata.ASCII_STRING_MARSHALLER), timestamp);
        headers.put(Metadata.Key.of(SIGNATURE_HEADER, Metadata.ASCII_STRING_MARSHALLER), signature);

        super.start(responseListener, headers);
      }
    };
  }

  private String encode(String secret, String message) {
    try {
      Mac mac = Mac.getInstance(ALGORITHM);
      mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), ALGORITHM));
      byte[] result = mac.doFinal(message.getBytes(StandardCharsets.UTF_8));
      return HexFormat.of().formatHex(result);
    } catch (Exception e) {
      throw new IllegalStateException("Failed to encode the authorization secret", e);
    }
  }
}