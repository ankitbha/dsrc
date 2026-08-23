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
        // Lets a unit test touch an Android stub without throwing. Nothing currently
        // relies on it -- removing it leaves every unit test green -- but it is a trap
        // worth knowing: under it Build.VERSION.SDK_INT reads 0, so any version-gated
        // branch silently takes the legacy path. Version-dependent behaviour belongs in
        // androidTest, where the platform is real.
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
    androidTestImplementation(libs.androidx.test.rules)
    androidTestImplementation(libs.androidx.test.core)
}

// Read the intended manifest facts back out of the *merged* manifest rather than
// restating them in a unit test. A test that repeats `minSdk = 29` from the same
// build file proves nothing; this fails if the merge produced something else, which
// is the only way the number can actually go wrong.
//
// Matching is on the full attribute including its closing quote. A `contains` on the
// bare class name passes for `.MainActivityX`, which is a ClassNotFoundException at
// launch -- exactly what reading the artifact back is supposed to catch.
val verifyMergedManifest by tasks.registering {
    val manifestDir = layout.buildDirectory.dir("intermediates/merged_manifest/debug")
    inputs.dir(manifestDir).withPropertyName("mergedManifest")
    outputs.upToDateWhen { false }

    doLast {
        val manifest = manifestDir.get().asFile.walkTopDown()
            .firstOrNull { it.name == "AndroidManifest.xml" }
            ?: error("merged manifest not found under ${manifestDir.get().asFile}")
        val text = manifest.readText()

        // Every permission the app cannot run without. FOREGROUND_SERVICE is the one
        // that bites hardest if it goes missing: startForeground() throws
        // SecurityException without it on every device at minSdk 29.
        val permissions = listOf(
            "android.permission.CAMERA",
            "android.permission.ACCESS_FINE_LOCATION",
            "android.permission.ACCESS_COARSE_LOCATION",
            "android.permission.INTERNET",
            "android.permission.FOREGROUND_SERVICE",
            "android.permission.FOREGROUND_SERVICE_CAMERA",
            "android.permission.FOREGROUND_SERVICE_LOCATION",
            "android.permission.POST_NOTIFICATIONS",
        )

        val expected = buildMap {
            put("android:minSdkVersion=\"29\"", "minSdk 29")
            put("android:targetSdkVersion=\"35\"", "targetSdk 35")
            put("android:foregroundServiceType=\"camera|location\"", "foreground-service type")
            put("android:name=\"com.dsrc.phone.SensingService\"", "sensing service")
            put("android:name=\"com.dsrc.phone.MainActivity\"", "launcher activity")

            permissions.forEach { put("android:name=\"$it\"", it.substringAfterLast('.')) }
        }

        val missing = expected.filterKeys { !text.contains(it) }.toMutableMap()

        // Attributes that must hold on one specific element are checked inside that
        // element. Searching the whole file for android:exported="false" passes on any
        // AGP-injected component that happens to carry it, while the service itself is
        // exported -- which would let any app on the phone start sensing.
        // Anchored on the closing quote, like the `expected` map. Without it a
        // <service android:name=".SensingServiceHelper"> matches first and the real
        // service's attributes go unchecked.
        val serviceElement = Regex("""<service\b[^>]*android:name="com\.dsrc\.phone\.SensingService"[^>]*>""")
            .find(text)?.value
        if (serviceElement == null) {
            missing["<service> element"] = "sensing service element"
        } else {
            if (!serviceElement.contains("android:exported=\"false\"")) {
                missing["service exported"] = "service exported=false"
            }
            // The type is already required by `expected`; here it is confirmed to sit
            // on this element rather than merely somewhere in the file.
            if (!serviceElement.contains("android:foregroundServiceType=\"camera|location\"")) {
                missing["service fgs type"] = "service foregroundServiceType on the service"
            }
        }

        if (missing.isNotEmpty()) {
            error("merged manifest is missing: ${missing.values.joinToString(", ")}\n  ${manifest.path}")
        }
        // expected.size file-wide facts, plus the two element-scoped ones.
        logger.lifecycle("merged manifest verified: ${expected.size} file facts + 2 on the service element, ${manifest.path}")
    }
}

// AGP registers its variant tasks after this script is evaluated, so the wiring has
// to wait for it. Without the dependency the verify task would run before any
// manifest existed and fail for the wrong reason.
afterEvaluate {
    verifyMergedManifest.configure { dependsOn("processDebugMainManifest") }
    tasks.named("check") { dependsOn(verifyMergedManifest) }
}

// The source manifest is an input to the unit tests too: ManifestPermissionsTest ties
// PermissionModel's constants to the strings actually declared. Same lesson as the
// protocol spec -- without declaring it, editing only the manifest leaves the test
// task UP-TO-DATE and the tie unchecked.
tasks.withType<Test>().configureEach {
    val sourceManifest = layout.projectDirectory.file("src/main/AndroidManifest.xml")
    inputs.file(sourceManifest).withPropertyName("sourceManifest")
    systemProperty("dsrc.manifest", sourceManifest.asFile.absolutePath)
}
