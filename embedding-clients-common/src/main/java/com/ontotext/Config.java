package com.ontotext;

public class Config {

    public static String getProperty(String key) {
        return System.getProperty(key);
    }

    public static String getProperty(String key, String defaultValue) {
        return System.getProperty(key, defaultValue);
    }

    public static int getPropertyInt(String key) {
        return parseInt(key, getProperty(key));
    }

    public static int getPropertyInt(String key, int defaultValue) {
        return parseInt(key, getProperty(key, String.valueOf(defaultValue)));
    }

    private static int parseInt(String key, String value) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalStateException(
                    "Illegal value " + value + ", for integer configuration parameter: " + key);
        }
    }

    public static boolean getPropertyBoolean(String key) {
        return parseBoolean(key, getProperty(key));
    }

    public static boolean getPropertyBoolean(String key, boolean defaultValue) {
        return parseBoolean(key, getProperty(key, String.valueOf(defaultValue)));
    }

    private static boolean parseBoolean(String key, String value) {
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
