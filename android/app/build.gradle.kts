plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "io.github.nikh93.gsplat"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.nikh93.gsplat"
        minSdk = 26 // Cardboard SDK
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
        ndk {
            // the Cardboard SDK ships these ABIs
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_static")
            }
        }
    }

    // JNI bridge of the Cardboard SDK (see cardboard/build_cardboard_sdk.sh)
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    packaging {
        jniLibs {
            // provided by the Cardboard AAR and referenced by the JNI bridge
            pickFirsts += "**/libGfxPluginCardboard.so"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("com.google.ar:core:1.46.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.recyclerview:recyclerview:1.3.2")
    implementation("androidx.swiperefreshlayout:swiperefreshlayout:1.1.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("com.google.android.material:material:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // Google Cardboard SDK, built from the official release by cardboard/build_cardboard_sdk.sh,
    // plus the dependencies of its AAR (QR code scanner, device parameter protos)
    implementation(files("../cardboard/cardboard-sdk.aar"))
    implementation("com.google.android.gms:play-services-vision:20.1.3")
    implementation("com.google.protobuf:protobuf-javalite:3.19.4")
    implementation("org.jspecify:jspecify:1.0.0")
}
