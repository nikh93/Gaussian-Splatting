# ARCore ships its own consumer rules; nothing app specific needed.

# Cardboard SDK Java API (called from native code)
-keep class com.google.cardboard.sdk.** { *; }
-keep class com.google.cardboard.proto.** { *; }
