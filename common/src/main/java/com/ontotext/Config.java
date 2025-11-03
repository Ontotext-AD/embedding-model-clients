package com.ontotext;

public class Config {

  public static String getProperty(String key) {
    return System.getProperty(key);
  }

  public static int getPropertyInt(String key) {
    String value = getProperty(key);
    try {
      return Integer.parseInt(value);
    } catch (NumberFormatException e) {
      throw new IllegalStateException(
              "Illegal value " + value + ", for integer configuration parameter: " + key);
    }
  }

  public static boolean getPropertyBoolean(String key) {
    String value = getProperty(key);

    if (value.equalsIgnoreCase("true") || value.equals("1") || value.equalsIgnoreCase("y")) {
      return true;
    }

    if (value.equalsIgnoreCase("false") || value.equals("0") || value.equalsIgnoreCase("n")) {
      return false;
    }

    throw new IllegalStateException(
            "Illegal value " + value + ", for boolean configuration parameter: " + key);
  }

  private Config() {
  }
}
