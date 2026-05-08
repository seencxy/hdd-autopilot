use std::path::PathBuf;
use std::process::Command;

struct WindowsOpenclPaths {
    include_dir: PathBuf,
    library_file: PathBuf,
    lib_dir: PathBuf,
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(mining_opencl_native_enabled)");
    println!("cargo:rerun-if-env-changed=HOST");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=OPENCL_ROOT");
    println!("cargo:rerun-if-env-changed=OpenCL_ROOT");
    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
    println!("cargo:rerun-if-env-changed=HDD_AUTOPILOT_DISABLE_OPENCL");
    println!("cargo:rerun-if-env-changed=HDD_AUTOPILOT_FORCE_OPENCL");
    println!("cargo:rerun-if-changed=../../native/mining-opencl/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../../native/mining-opencl/include");
    println!("cargo:rerun-if-changed=../../native/mining-opencl/src");
    println!(
        "cargo:rerun-if-changed=../../native/mining-opencl/third_party/argon2-gpu/include/argon2-opencl"
    );
    println!(
        "cargo:rerun-if-changed=../../native/mining-opencl/third_party/argon2-gpu/src/argon2-opencl"
    );

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" && target_os != "linux" && target_os != "windows" {
        return;
    }
    if env_flag("HDD_AUTOPILOT_DISABLE_OPENCL") {
        warn_native_disabled("HDD_AUTOPILOT_DISABLE_OPENCL is set");
        return;
    }

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_os == "windows" && target_arch != "x86_64" {
        warn_native_disabled(
            "target architecture is not supported for Windows OpenCL native backend",
        );
        return;
    }

    let cmake_arch = if target_os == "macos" {
        let Some(cmake_arch) = macos_cmake_arch(&target_arch) else {
            warn_native_disabled(
                "target architecture is not supported for macOS OpenCL native backend",
            );
            return;
        };
        Some(cmake_arch)
    } else {
        None
    };

    if target_os == "macos" && !host_is_macos() {
        warn_native_disabled("host is not macOS; skipping OpenCL native backend");
        return;
    }

    if target_os == "linux" && !host_is_matching_linux(&target_arch) {
        warn_native_disabled("Linux host architecture does not match OpenCL target architecture");
        return;
    }

    if target_os == "windows" && !host_is_windows() {
        warn_native_disabled("host is not Windows; skipping OpenCL native backend");
        return;
    }
    if target_os == "windows"
        && !env_flag("HDD_AUTOPILOT_FORCE_OPENCL")
        && windows_cuda_toolkit_present()
    {
        warn_native_disabled(
            "CUDA Toolkit detected; skipping Windows OpenCL native backend to avoid duplicate native solver symbols",
        );
        return;
    }

    let windows_opencl_paths = if target_os == "windows" {
        prepare_windows_build_environment();
        find_windows_opencl_paths()
    } else {
        None
    };

    let Some(dst) = build_native(cmake_arch, windows_opencl_paths.as_ref()) else {
        warn_native_disabled(
            "OpenCL native backend build failed; falling back to unavailable backend",
        );
        return;
    };

    let lib_dir = dst.join("build").join("lib");
    println!("cargo:rustc-cfg=mining_opencl_native_enabled");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=mining_opencl_core");
    println!("cargo:rustc-link-lib=static=argon2_ref");
    println!("cargo:rustc-link-lib=static=argon2_opencl");
    println!("cargo:rustc-link-lib=static=argon2_gpu_common");
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=OpenCL");
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        if let Some(paths) = windows_opencl_paths.as_ref() {
            println!("cargo:rustc-link-search=native={}", paths.lib_dir.display());
        }
        println!("cargo:rustc-link-lib=dylib=OpenCL");
        if target_os == "linux" {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }
}

fn build_native(
    cmake_arch: Option<&str>,
    windows_opencl_paths: Option<&WindowsOpenclPaths>,
) -> Option<std::path::PathBuf> {
    run_with_suppressed_panic_output(|| {
        let mut config = cmake::Config::new("../../native/mining-opencl");
        config.profile("Release");
        if let Some(cmake_arch) = cmake_arch {
            config.define("CMAKE_OSX_ARCHITECTURES", cmake_arch);
        }
        if let Some(paths) = windows_opencl_paths {
            config.define("OpenCL_INCLUDE_DIR", &paths.include_dir);
            config.define("OpenCL_LIBRARY", &paths.library_file);
        }
        if let Some(make_program) = std::env::var_os("CMAKE_MAKE_PROGRAM") {
            config.define("CMAKE_MAKE_PROGRAM", PathBuf::from(make_program));
        }
        config.build_target("mining_opencl_core");
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
    println!("cargo:warning=OpenCL native backend disabled: {reason}");
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn windows_cuda_toolkit_present() -> bool {
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDAToolkit_ROOT"] {
        if let Some(root) = std::env::var_os(var)
            && cuda_root_has_nvcc(PathBuf::from(root))
        {
            return true;
        }
    }

    let cuda_root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    if !cuda_root.is_dir() {
        return false;
    }
    std::fs::read_dir(cuda_root)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .any(cuda_root_has_nvcc)
}

fn cuda_root_has_nvcc(root: PathBuf) -> bool {
    root.join("bin").join("nvcc.exe").is_file()
}

fn prepare_windows_build_environment() {
    if let Some(vs_env) = load_vs_dev_env() {
        for (key, value) in vs_env {
            set_env_var(key, value);
        }
    }
    if let Some(cmake_path) = find_visual_studio_cmake() {
        set_env_var("CMAKE", cmake_path);
    }
    if let Some(path) = find_visual_studio_ninja() {
        set_env_var("CMAKE_GENERATOR", "Ninja");
        set_env_var("CMAKE_MAKE_PROGRAM", path);
    }
}

fn set_env_var<K, V>(key: K, value: V)
where
    K: AsRef<std::ffi::OsStr>,
    V: AsRef<std::ffi::OsStr>,
{
    unsafe {
        std::env::set_var(key, value);
    }
}

fn host_is_macos() -> bool {
    std::env::var("HOST")
        .map(|host| host.contains("apple-darwin"))
        .unwrap_or(false)
}

fn host_is_windows() -> bool {
    std::env::var("HOST")
        .map(|host| host.contains("windows"))
        .unwrap_or(false)
}

fn host_is_matching_linux(target_arch: &str) -> bool {
    let Ok(host) = std::env::var("HOST") else {
        return false;
    };
    if !host.contains("linux") {
        return false;
    }
    match target_arch {
        "x86_64" => host.starts_with("x86_64-"),
        "aarch64" => host.starts_with("aarch64-"),
        _ => false,
    }
}

fn macos_cmake_arch(target_arch: &str) -> Option<&'static str> {
    match target_arch {
        "x86_64" => Some("x86_64"),
        "aarch64" => Some("arm64"),
        _ => None,
    }
}

fn find_windows_opencl_paths() -> Option<WindowsOpenclPaths> {
    let mut roots = Vec::new();
    for var in ["OPENCL_ROOT", "OpenCL_ROOT", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(var) {
            roots.push(PathBuf::from(root));
        }
    }
    if let Some(vcpkg_root) = std::env::var_os("VCPKG_ROOT") {
        roots.push(
            PathBuf::from(vcpkg_root)
                .join("installed")
                .join("x64-windows"),
        );
    }
    let cuda_root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    if cuda_root.is_dir()
        && let Ok(entries) = std::fs::read_dir(cuda_root)
    {
        roots.extend(
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_dir()),
        );
    }

    roots.into_iter().find_map(|root| {
        let include_dir = root.join("include");
        let library_file = root.join("lib").join("x64").join("OpenCL.lib");
        if include_dir.join("CL").join("opencl.h").is_file() && library_file.is_file() {
            let lib_dir = library_file.parent()?.to_path_buf();
            Some(WindowsOpenclPaths {
                include_dir,
                library_file,
                lib_dir,
            })
        } else {
            None
        }
    })
}

fn load_vs_dev_env() -> Option<Vec<(String, String)>> {
    let install = find_vs_installation()?;
    let vsdevcmd = install.join("Common7").join("Tools").join("VsDevCmd.bat");
    if !vsdevcmd.is_file() {
        return None;
    }
    let command = format!(
        "call \"{}\" -arch=x64 -host_arch=x64 >nul && set",
        vsdevcmd.display()
    );
    let output = Command::new("cmd")
        .arg("/d")
        .arg("/c")
        .arg(command)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let vars = text
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.trim().to_string(), value.trim().to_string()))
        .collect::<Vec<_>>();
    if vars.is_empty() { None } else { Some(vars) }
}

fn find_visual_studio_cmake() -> Option<PathBuf> {
    let install = find_vs_installation()?;
    let path = install
        .join("Common7")
        .join("IDE")
        .join("CommonExtensions")
        .join("Microsoft")
        .join("CMake")
        .join("CMake")
        .join("bin")
        .join("cmake.exe");
    path.is_file().then_some(path)
}

fn find_visual_studio_ninja() -> Option<PathBuf> {
    let install = find_vs_installation()?;
    let path = install
        .join("Common7")
        .join("IDE")
        .join("CommonExtensions")
        .join("Microsoft")
        .join("CMake")
        .join("Ninja")
        .join("ninja.exe");
    path.is_file().then_some(path)
}

fn find_vs_installation() -> Option<PathBuf> {
    let vswhere =
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    if !vswhere.is_file() {
        return None;
    }
    let output = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let install = String::from_utf8(output.stdout).ok()?;
    let trimmed = install.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}
