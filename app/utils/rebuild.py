"""
静态页面重建工具
用于在内容更新后自动重建 Astro 静态页面
支持零停机构建（原子替换）
"""
import os
import shutil
import subprocess
import threading
import time
import platform
from typing import Optional

# 重建状态
_rebuild_lock = threading.Lock()
_is_rebuilding = False
_last_trigger_time: float = 0
_pending_rebuild = False

# 缓存的 frontend 路径
_cached_frontend_dir: Optional[str] = None

# 防抖动时间（秒）- 短时间内多次触发只执行一次
DEBOUNCE_SECONDS = 2.0

# 目录名称
DIST_DIR = "dist"
DIST_NEW_DIR = "dist_new"
DIST_OLD_DIR = "dist_old"


def _get_frontend_dir() -> str:
    """获取并缓存 frontend 目录路径"""
    global _cached_frontend_dir
    if _cached_frontend_dir is None:
        fd = os.path.join(os.getcwd(), "frontend")
        _cached_frontend_dir = fd if os.path.exists(fd) else "frontend"
    return _cached_frontend_dir


def _atomic_swap_dirs(frontend_dir: str) -> bool:
    """
    原子替换目录：dist_new -> dist

    步骤：
    1. 如果存在 dist_old，删除它
    2. 如果存在 dist，重命名为 dist_old
    3. 将 dist_new 重命名为 dist
    4. 删除 dist_old

    Returns:
        True 如果替换成功，False 如果失败
    """
    dist_path = os.path.join(frontend_dir, DIST_DIR)
    dist_new_path = os.path.join(frontend_dir, DIST_NEW_DIR)
    dist_old_path = os.path.join(frontend_dir, DIST_OLD_DIR)

    try:
        # 检查 dist_new 是否存在
        if not os.path.exists(dist_new_path):
            print(f"[Rebuild] ❌ 新构建目录不存在: {dist_new_path}")
            return False

        # 1. 删除旧的 dist_old（如果存在）
        if os.path.exists(dist_old_path):
            shutil.rmtree(dist_old_path)
            print("[Rebuild] 🗑️ 已删除旧的 dist_old")

        # 2. 将当前 dist 重命名为 dist_old
        if os.path.exists(dist_path):
            os.rename(dist_path, dist_old_path)
            print("[Rebuild] 📦 dist -> dist_old")

        # 3. 将 dist_new 重命名为 dist（原子操作）
        os.rename(dist_new_path, dist_path)
        print("[Rebuild] 🔄 dist_new -> dist (原子替换完成)")

        # 4. 异步删除 dist_old（不阻塞）
        if os.path.exists(dist_old_path):
            def cleanup():
                try:
                    shutil.rmtree(dist_old_path)
                    print("[Rebuild] 🗑️ 已清理 dist_old")
                except Exception as e:
                    print(f"[Rebuild] ⚠️ 清理 dist_old 失败: {e}")

            cleanup_thread = threading.Thread(target=cleanup, daemon=True)
            cleanup_thread.start()

        return True

    except Exception as e:
        print(f"[Rebuild] ❌ 目录替换失败: {e}")
        # 尝试恢复
        try:
            if not os.path.exists(dist_path) and os.path.exists(dist_old_path):
                os.rename(dist_old_path, dist_path)
                print("[Rebuild] 🔙 已恢复原 dist 目录")
        except Exception as restore_error:
            print(f"[Rebuild] ❌ 恢复失败: {restore_error}")
        return False


def trigger_rebuild_async(timeout: int = 300):
    """
    在后台线程中触发前端静态页面重建（带防抖动）

    Args:
        timeout: 构建超时时间（秒），默认 300 秒
    """
    global _last_trigger_time, _pending_rebuild

    current_time = time.time()

    # 防抖动：如果距离上次触发时间太短，标记为待处理
    with _rebuild_lock:
        if current_time - _last_trigger_time < DEBOUNCE_SECONDS:
            _pending_rebuild = True
            print("[Rebuild] 📋 防抖动：已标记待重建")
            return
        _last_trigger_time = current_time
        _pending_rebuild = False

    # 立即启动后台线程
    thread = threading.Thread(target=_run_build, args=(timeout,), daemon=True)
    thread.start()
    print("[Rebuild] 📋 重建任务已加入队列")


def _run_build(timeout: int = 300):
    """实际执行构建的函数（零停机版本）"""
    global _is_rebuilding, _pending_rebuild, _last_trigger_time

    with _rebuild_lock:
        if _is_rebuilding:
            print("[Rebuild] ⏳ 已有构建任务在运行，跳过")
            return
        _is_rebuilding = True

    try:
        fd = _get_frontend_dir()
        dist_new_path = os.path.join(fd, DIST_NEW_DIR)

        # 清理可能存在的旧 dist_new 目录
        if os.path.exists(dist_new_path):
            shutil.rmtree(dist_new_path)

        print(f"[Rebuild] 🚀 开始构建静态页面... ({fd})")
        print(f"[Rebuild] 📁 构建输出目录: {DIST_NEW_DIR}")

        # 构建命令：输出到 dist_new 目录
        is_windows = platform.system() == "Windows"
        # Astro 使用 --outDir 参数指定输出目录
        if is_windows:
            cmd = ["npm", "run", "build", "--", "--outDir", DIST_NEW_DIR]
        else:
            cmd = f"npm run build -- --outDir {DIST_NEW_DIR}"

        process = subprocess.Popen(
            cmd,
            cwd=fd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode == 0:
                print("[Rebuild] ✅ 构建完成，开始原子替换...")

                # 执行原子替换
                if _atomic_swap_dirs(fd):
                    print("[Rebuild] ✅ 静态页面更新成功（零停机）")
                else:
                    print("[Rebuild] ❌ 目录替换失败")
            else:
                # 优先显示 stderr，如果为空则显示 stdout
                error_msg = stderr.strip() if stderr and stderr.strip() else stdout.strip() if stdout else 'Unknown error'
                print(f"[Rebuild] ❌ 构建失败 (code={process.returncode}):")
                print(f"[Rebuild] 错误信息: {error_msg[:1000]}")

                # 清理失败的构建目录
                if os.path.exists(dist_new_path):
                    shutil.rmtree(dist_new_path)

        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            print("[Rebuild] ⚠️ 构建超时")

            # 清理超时的构建目录
            if os.path.exists(dist_new_path):
                shutil.rmtree(dist_new_path)

    except Exception as e:
        print(f"[Rebuild] ❌ 构建错误: {e}")
    finally:
        with _rebuild_lock:
            _is_rebuilding = False

        # 检查是否有待处理的重建请求
        if _pending_rebuild:
            with _rebuild_lock:
                _pending_rebuild = False
                _last_trigger_time = time.time()
            print("[Rebuild] 🔄 处理待定的重建请求...")
            thread = threading.Thread(target=_run_build, args=(timeout,), daemon=True)
            thread.start()


def is_rebuilding() -> bool:
    """检查当前是否正在重建"""
    with _rebuild_lock:
        return _is_rebuilding


def get_rebuild_status() -> dict:
    """
    检查最近一次构建的状态。
    通过检查 dist 目录的修改时间来判断。

    Returns:
        包含构建状态信息的字典
    """
    fd = _get_frontend_dir()
    dist_dir = os.path.join(fd, DIST_DIR)

    if os.path.exists(dist_dir):
        mtime = os.path.getmtime(dist_dir)
        last_build = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        return {
            "status": "success",
            "is_rebuilding": is_rebuilding(),
            "last_build": last_build,
            "message": f"最近构建时间: {last_build}"
        }
    else:
        return {
            "status": "not_built",
            "is_rebuilding": is_rebuilding(),
            "message": "尚未构建静态页面，请先运行 npm run build"
        }


def run_rebuild_sync(timeout: int = 300) -> dict:
    """
    同步执行重建（阻塞式，零停机版本）

    Args:
        timeout: 构建超时时间（秒）

    Returns:
        包含构建结果的字典
    """
    try:
        fd = _get_frontend_dir()
        dist_new_path = os.path.join(fd, DIST_NEW_DIR)

        # 清理可能存在的旧 dist_new 目录
        if os.path.exists(dist_new_path):
            shutil.rmtree(dist_new_path)

        print(f"[Rebuild] 🚀 开始同步构建静态页面... ({fd})")

        # 构建命令：输出到 dist_new 目录
        is_windows = platform.system() == "Windows"
        if is_windows:
            cmd = ["npm", "run", "build", "--", "--outDir", DIST_NEW_DIR]
        else:
            cmd = f"npm run build -- --outDir {DIST_NEW_DIR}"

        result = subprocess.run(
            cmd,
            cwd=fd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )

        if result.returncode == 0:
            print("[Rebuild] ✅ 构建完成，开始原子替换...")

            if _atomic_swap_dirs(fd):
                print("[Rebuild] ✅ 静态页面更新成功（零停机）")
                return {
                    "status": "success",
                    "message": "构建成功（零停机更新）"
                }
            else:
                return {
                    "status": "failed",
                    "message": "构建成功但目录替换失败"
                }
        else:
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip() if result.stdout else 'Unknown error'
            print(f"[Rebuild] ❌ 构建失败: {error_msg[:300]}")

            # 清理失败的构建目录
            if os.path.exists(dist_new_path):
                shutil.rmtree(dist_new_path)

            return {
                "status": "failed",
                "message": f"构建失败: {error_msg[:300]}"
            }

    except subprocess.TimeoutExpired:
        print("[Rebuild] ⚠️ 构建超时")
        return {
            "status": "timeout",
            "message": "构建超时"
        }
    except Exception as e:
        print(f"[Rebuild] ❌ 构建错误: {e}")
        return {
            "status": "error",
            "message": f"构建错误: {str(e)}"
        }
