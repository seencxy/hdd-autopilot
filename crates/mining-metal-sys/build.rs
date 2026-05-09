fn main() {
    println!("cargo:rustc-check-cfg=cfg(mining_metal_native_enabled)");
    println!("cargo:rerun-if-env-changed=HOST");
    println!("cargo:rerun-if-changed=../../native/mining-metal/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../../native/mining-metal/include");
    println!("cargo:rerun-if-changed=../../native/mining-metal/src");
    println!("cargo:rerun-if-changed=../../native/mining-opencl/third_party/argon2/include");
    println!("cargo:rerun-if-changed=../../native/mining-opencl/third_party/argon2/src");
    println!(
        "cargo:rerun-if-changed=../../native/mining-opencl/third_party/argon2-gpu/include/argon2-gpu-common"
    );
    println!(
        "cargo:rerun-if-changed=../../native/mining-opencl/third_party/argon2-gpu/src/argon2-gpu-common"
    );

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let Some(cmake_arch) = macos_cmake_arch(&target_arch) else {
        warn_native_disabled("当前目标架构不是受支持的 macOS 架构，跳过 Metal 原生后端构建。");
        return;
    };

    if !host_is_macos() {
        warn_native_disabled("当前不是 macOS 宿主，跳过 Metal 原生后端构建。");
        return;
    }

    if !cmake_is_available() {
        warn_native_disabled("找不到 cmake 命令；请先安装 CMake，例如运行 brew install cmake。");
        return;
    }

    let Some(dst) = build_native(cmake_arch) else {
        warn_native_disabled("Metal 原生后端构建失败，已自动降级为不可用后端。");
        return;
    };

    let lib_dir = dst.join("build").join("lib");
    println!("cargo:rustc-cfg=mining_metal_native_enabled");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=mining_metal_core");
    println!("cargo:rustc-link-lib=static=argon2_ref");
    println!("cargo:rustc-link-lib=static=argon2_gpu_common");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=dylib=c++");
}

fn build_native(cmake_arch: &str) -> Option<std::path::PathBuf> {
    run_with_suppressed_panic_output(|| {
        let mut config = cmake::Config::new("../../native/mining-metal");
        config
            .profile("Release")
            .define("CMAKE_OSX_ARCHITECTURES", cmake_arch)
            .build_target("mining_metal_core");
        config.build()
    })
}

fn run_with_suppressed_panic_output<T, F>(f: F) -> Option<T>
where
    F: FnOnce() -> T,
{
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok();
    std::panic::set_hook(hook);
    result
}

fn warn_native_disabled(reason: &str) {
    println!("cargo:warning=Metal native backend disabled: {reason}");
}

fn host_is_macos() -> bool {
    std::env::var("HOST")
        .map(|host| host.contains("apple-darwin"))
        .unwrap_or(false)
}

fn cmake_is_available() -> bool {
    std::process::Command::new("cmake")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn macos_cmake_arch(target_arch: &str) -> Option<&'static str> {
    match target_arch {
        "x86_64" => Some("x86_64"),
        "aarch64" => Some("arm64"),
        _ => None,
    }
}
