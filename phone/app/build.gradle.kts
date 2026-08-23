plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.dsrc.phone"
    // Only platform 35 is installed on the build machine; this is not a preference.
    compileSdk = 35

    defaultConfig {
        applicationId = "com.dsrc.phone"
        // 29 covers CameraX, typed foreground services and everything else used
        // here, and sits below the API-31 system image the tests run on. It is a
        // guess about the handset -- see plan_task17_android_skeleton.md O1.
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    sourceSets {
        getByName("main").java.srcDirs("src/main/kotlin")
        getByName("test").java.srcDirs("src/test/kotlin")
        getByName("androidTest").java.srcDirs("src/androidTest/kotlin")
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    implementation(project(":transport"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.lifecycle.service)

    // Declared now so the dependency resolves and the toolchain is proven; the
    // camera is wired up in task 18.
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.test.junit)
    androidTestImplementation(libs.espresso.core)
}

// Read the intended manifest facts back out of the *merged* manifest rather than
// restating them in a unit test. A test that repeats `minSdk = 29` from the same
// build file proves nothing; this fails if the merge produced something else, which
// is the only way the number can actually go wrong.
//
// Wired into `check` and declared as an input on the manifest task, so it cannot
// pass by running before the manifest exists.
val verifyMergedManifest by tasks.registering {
    val manifestDir = layout.buildDirectory.dir("intermediates/merged_manifest/debug")
    inputs.dir(manifestDir).withPropertyName("mergedManifest")
    outputs.upToDateWhen { false }

    doLast {
        val manifest = manifestDir.get().asFile.walkTopDown()
            .firstOrNull { it.name == "AndroidManifest.xml" }
            ?: error("merged manifest not found under ${manifestDir.get().asFile}")
        val text = manifest.readText()

        val expected = mapOf(
            "android:minSdkVersion=\"29\"" to "minSdk",
            "android:targetSdkVersion=\"35\"" to "targetSdk",
            "android:foregroundServiceType=\"camera|location\"" to "foreground-service type",
            "android.permission.FOREGROUND_SERVICE_CAMERA" to "camera FGS permission",
            "android.permission.FOREGROUND_SERVICE_LOCATION" to "location FGS permission",
            "android.permission.POST_NOTIFICATIONS" to "notification permission",
            "com.dsrc.phone.SensingService" to "sensing service",
            "com.dsrc.phone.MainActivity" to "launcher activity",
        )
        val missing = expected.filterKeys { !text.contains(it) }
        if (missing.isNotEmpty()) {
            error("merged manifest is missing: ${missing.values.joinToString(", ")}\n  ${manifest.path}")
        }
        logger.lifecycle("merged manifest verified: ${expected.size} facts, ${manifest.path}")
    }
}

// AGP registers its variant tasks after this script is evaluated, so the wiring has
// to wait for it. Without the dependency the verify task would run before any
// manifest existed and fail for the wrong reason.
afterEvaluate {
    verifyMergedManifest.configure { dependsOn("processDebugMainManifest") }
    tasks.named("check") { dependsOn(verifyMergedManifest) }
}
