"""UI translations. Language never changes generation parameters or presets."""
from urllib.error import HTTPError, URLError

EN = {
    "stage": "Insertion point",
    "before_hr": "Before Hires. fix (after first pass if Hires is off)",
    "after_hr": "After Hires. fix",
    "behavior": "NR runs before face restoration / ADetailer. Failure stops generation; the original is not reported as enhanced. No automatic GPU switching or unloading of other models.",
    "style": "Style index (0-2)",
    "preset": "Internal model preset (0-3)",
    "intensity": "Intensity",
    "tone": "Local tone",
    "structure": "Local structure",
    "skin": "Skin structure",
    "mix": "Output blend (0 = original, 1 = NR; the model still runs)",
    "auto_mask": "Automatic mask",
    "flow": "Temporal optical flow (not used for still images)",
    "experimental": "Model controls are experimental, not calibrated quality levels. Skin defaults to -1. Every still image resets temporal history.",
    "presets": "Named parameter presets",
    "preset_scope": "Presets store parameters and insertion point, not the enable switch, GPU or runtime. Saving the same name replaces it; deleting only removes that preset.",
    "saved": "Saved preset",
    "load": "Load parameters",
    "delete": "Delete selected preset",
    "name": "Preset name",
    "save": "Save / replace preset",
    "preset_status": "Preset status",
    "environment_group": "Choose a GPU",
    "ownership": "Forge starts its own private NR worker. The worker exits with Forge, and no separate server port is opened.",
    "environment_note": "Choose the NVIDIA adapter by its model name. No model is loaded until generation.",
    "device": "NR GPU (explicit physical adapter)",
    "check": "Check runtime",
    "enumerate": "List GPUs",
    "runtime_json": "Runtime snapshot JSON",
    "environment": "Readiness",
    "environment_waiting": "Waiting for the page to check runtime files. No device has been enumerated or model loaded.",
    "status": "Refresh worker status",
    "release": "Release idle NR resources",
    "worker_status": "Worker status",
    "status_waiting": "Not queried. Releasing resources is refused while NR is processing a request.",
    "read_presets_failed": "Could not read preset file: {error}",
    "not_ready": "Runtime is not ready",
    "device_mismatch": "The worker mapping does not match the selected GPU; automatic fallback is disabled.",
    "files_only": "Runtime files checked only. Hardware and NR execution are verified during generation.",
    "environment_failed": "Runtime error (selection retained; automatic fallback disabled): {error}",
    "no_devices": "No NVIDIA adapter with a verified CUDA/LUID mapping was found.",
    "device_missing": "The selected GPU was not returned by enumeration. Check the physical adapter mapping.",
    "enumerated": "GPU detection finished. Select the NVIDIA adapter to use from the list.",
    "enumerate_failed": "GPU enumeration failed (selection unchanged): {error}",
    "private": "Private Forge NR (closed with Forge)",
    "released": "Idle resources released.\n",
    "worker_failed": "Worker operation failed: {error}",
    "saved_ok": "Parameters and insertion point saved. Enable switch and runtime were not saved.",
    "save_failed": "Save failed: {error}",
    "loaded_ok": "Parameters loaded. Enable switch and runtime are unchanged.",
    "hires_reset": " Hires. fix is off, so insertion is set to before the hires pass.",
    "load_failed": "Load failed: {error}",
    "deleted_ok": "Preset deleted. Current parameters are unchanged.",
    "delete_failed": "Delete failed: {error}",
    "setup": "First-time setup",
    "setup_status": "Preparation status",
    "runtime_mode": "Runtime source",
    "runtime_auto": "Automatic original",
    "runtime_manual": "Manual custom / community",
    "runtime_location": "Current runtime directory",
    "runtime_auto_note": "Downloads only the NVIDIA-original member from the configured third-party release. NVIDIA signature verification remains enabled.",
    "runtime_manual_note": "Manual mode never downloads or replaces your runtime. Only local DLL format is checked, not NVIDIA authenticity. Use files you trust; RTX 40/community compatibility has not been tested here.",
    "manual_files_only": "Manual runtime format checked; publisher signature and GPU compatibility are unverified.",
    "manual_missing": "Manual mode needs nvngx_dlssnr.dll in the displayed directory. Use Import runtime copy under Advanced, or place your file there yourself. No download was attempted.",
    "manual_invalid": "The selected manual file is not a usable Windows x64 DLL. Select the correct runtime; no file was installed.",
    "mode_invalid": "The saved runtime source is invalid. Check Diagnostic details; no automatic fallback was selected.",
    "runtime_file": "Existing NR runtime file (.dll)",
    "import_runtime": "Import runtime copy",
    "manual_setup": "Advanced: diagnostics and repair",
    "prepare": "Retry runtime and detect GPUs",
    "repair_dependencies": "Repair dependencies and continue",
    "repair_note": "Changing runtimes, importing files and repairing dependencies close only this extension's idle controller. Active NR tasks are not interrupted.",
    "setup_details": "Diagnostic details",
    "downloading": "Downloading NR runtime",
    "preparing": "Preparing runtime files. A missing runtime is downloaded automatically when a source is configured.",
    "source_missing": "Development build: the runtime download source has not been configured yet. Automatic installation cannot finish.",
    "prepare_failed": "Preparation did not finish. Check Diagnostic details under Advanced, then retry the indicated step.",
    "dependencies_missing": "Required dependencies are missing or damaged. Open Advanced: diagnostics and repair, then choose Repair dependencies and continue.",
    "dependencies_failed": "Dependency repair failed. Check your connection, then retry Repair dependencies and continue. Diagnostic details are under Advanced.",
    "repair_blocked": "NR is working or its idle state could not be confirmed. The runtime change or repair was not started. Wait for the task; restart Forge if the status remains unavailable.",
    "download_failed": "Runtime download did not finish. Check your connection and choose Retry runtime and detect GPUs.",
    "download_unavailable": "The runtime download is unavailable. Contact the maintainer; reinstalling Forge will not fix the source.",
    "download_invalid": "The runtime download was incomplete or invalid. Retry the runtime download; contact the maintainer if it fails again.",
    "source_invalid": "The runtime download configuration is invalid. Contact the maintainer; no system changes are needed.",
    "storage_failed": "The runtime file could not be saved. Check free space and access to Forge's model directory; your existing files were kept.",
    "choose_gpu": "Choose List GPUs, then select your NVIDIA adapter by model name.",
    "prepared": "Runtime files are ready. Enable DLSS5 NR and generate one image to test compatibility.",
    "ready_gpu": "Ready to generate with {gpu}.",
}

ZH = {
    "stage": "插入时机",
    "before_hr": "高清修复前（未开高清：首轮后）",
    "after_hr": "高清修复后",
    "behavior": "NR在修脸/ADetailer之前逐张处理。失败会中止本次生成，不保存原图冒充增强；不自动换卡或卸载其他模型。",
    "style": "风格编号（0-2）",
    "preset": "模型内部档位（0-3）",
    "intensity": "模型强度",
    "tone": "色调",
    "structure": "结构",
    "skin": "皮肤",
    "mix": "原图混合（0=原件，1=纯NR；仍执行模型）",
    "auto_mask": "自动蒙版",
    "flow": "时序光流（单图不使用）",
    "experimental": "模型控制项仍属实验性，不代表已校准的质量等级。皮肤默认-1，每张静图重置时序历史。",
    "presets": "本地命名参数预设",
    "preset_scope": "只保存参数与插入时机，不保存启用开关/显卡/环境。保存同名会覆盖；删除只删该条预设。",
    "saved": "已保存预设",
    "load": "载入参数",
    "delete": "删除选中预设",
    "name": "预设名",
    "save": "保存/覆盖当前参数",
    "preset_status": "预设状态",
    "environment_group": "选择显卡",
    "ownership": "Forge启动自己的NR私有进程，随Forge退出；不另开服务端口。",
    "environment_note": "按显卡型号选择要用的NVIDIA显卡；真正生成时才加载NR模型。",
    "device": "显式选择NR物理显卡",
    "check": "检查环境快照",
    "enumerate": "列出显卡",
    "runtime_json": "运行环境JSON快照",
    "environment": "就绪状态",
    "environment_waiting": "等待页面检查运行库；未枚举设备、未运行模型。",
    "status": "查询后台状态",
    "release": "释放空闲NR资源",
    "worker_status": "后台状态",
    "status_waiting": "尚未查询。NR正在处理任务时拒绝释放资源。",
    "read_presets_failed": "预设文件读取失败：{error}",
    "not_ready": "环境未就绪",
    "device_mismatch": "后台映射与所选显卡不一致，拒绝自动回退。",
    "files_only": "仅核对环境文件；硬件与NR执行在生成时验证。",
    "environment_failed": "环境错误（保留所选显卡，禁止隐式回退）：{error}",
    "no_devices": "没有具有物理CUDA/LUID映射的NVIDIA显卡。",
    "device_missing": "已选显卡不在本次物理枚举结果中，请检查设备映射。",
    "enumerated": "显卡检测完成，请在下拉框中按型号选择要用的NVIDIA显卡。",
    "enumerate_failed": "枚举失败（未改变选择）：{error}",
    "private": "Forge私有NR（随Forge关闭）",
    "released": "空闲资源已释放。\n",
    "worker_failed": "后台操作失败：{error}",
    "saved_ok": "已保存参数与时机；开关和环境未保存。",
    "save_failed": "保存失败：{error}",
    "loaded_ok": "已载入参数；开关与环境不变。",
    "hires_reset": " 未开高清，插入时机已设为高清前。",
    "load_failed": "载入失败：{error}",
    "deleted_ok": "已删除选中预设；当前参数不变。",
    "delete_failed": "删除失败：{error}",
    "setup": "首次准备",
    "setup_status": "准备状态",
    "runtime_mode": "运行库来源",
    "runtime_auto": "自动下载原版",
    "runtime_manual": "手动自定义／社区版",
    "runtime_location": "当前运行库目录",
    "runtime_auto_note": "从已配置的第三方发布包中仅下载NVIDIA原版成员，并检查NVIDIA签名。",
    "runtime_manual_note": "手动模式不会下载或覆盖你的运行库，只检查本地DLL格式，不验证NVIDIA签名。请使用可信文件；本项目尚未实测40系／社区版兼容性。",
    "manual_files_only": "手动文件格式已检查；未验证发布者签名或显卡兼容性。",
    "manual_missing": "手动模式缺少nvngx_dlssnr.dll。请在高级选项中导入副本，或自行放到上面显示的目录；没有尝试自动下载。",
    "manual_invalid": "手动文件不是可用的Windows x64 DLL，请选择正确的运行库；本次没有安装文件。",
    "mode_invalid": "保存的运行库来源无效，请查看诊断信息；没有自动改用其它来源。",
    "runtime_file": "已有NR运行库文件（.dll）",
    "import_runtime": "导入运行库副本",
    "manual_setup": "高级：诊断与修复",
    "prepare": "重试运行库并检测显卡",
    "repair_dependencies": "修复依赖并继续",
    "repair_note": "切换运行库、导入和修复依赖只会关闭本插件的空闲后台，正在执行的NR任务不会被打断。",
    "setup_details": "诊断信息",
    "downloading": "正在下载NR运行库",
    "preparing": "正在准备运行环境；配置下载源后，缺少的运行库会自动下载。",
    "source_missing": "开发版尚未配置运行库下载源，自动安装暂时不能完成。",
    "prepare_failed": "准备未完成。请展开“高级：诊断与修复”查看诊断信息，再重试对应步骤。",
    "dependencies_missing": "缺少依赖或依赖已损坏。请展开“高级：诊断与修复”，点击“修复依赖并继续”。",
    "dependencies_failed": "依赖修复未完成。请检查网络后再次点击“修复依赖并继续”；详细原因在高级选项的诊断信息中。",
    "repair_blocked": "NR正在处理任务，或暂时无法确认后台空闲，未开始切换或修复。请等待任务结束；持续无法查询时重启Forge再试。",
    "download_failed": "运行库下载未完成。请检查网络，再点击“重试运行库并检测显卡”。",
    "download_unavailable": "运行库下载文件暂时不可用，请联系项目维护者；无需重装Forge。",
    "download_invalid": "下载的运行库不完整或格式有误。请重试运行库下载；重复失败时联系项目维护者。",
    "source_invalid": "项目的运行库下载配置无效，请联系维护者；无需修改电脑设置。",
    "storage_failed": "运行库无法保存。请检查剩余空间和Forge模型目录的访问权限；已有文件未被覆盖。",
    "choose_gpu": "点击“列出显卡”，再按型号选择自己的NVIDIA显卡。",
    "prepared": "运行库已准备好，开启DLSS5 NR后生成一张图片验证兼容性。",
    "ready_gpu": "已准备好，可以使用{gpu}生成。",
}

CATALOGS = {"en": EN, "zh": ZH}
ERRORS_EN = {
    "缺少可用的 NR bridge": "The NR bridge is missing or unusable",
    "缺少用户自行提供并获授权的 NVIDIA nvngx_dlssnr.dll": "A licensed, user-supplied NVIDIA nvngx_dlssnr.dll is required",
    "缺少 NR caller/nvngx.dll_comfy.dll": "The MIT NR caller helper is missing",
    "NR bridge 缺少处理/物理设备核对接口，需要项目构建的 MIT bridge": "The bridge is missing NR/physical-device verification exports; install this project's MIT bridge",
    "NR bridge 构建清单无法读取": "The bridge build manifest could not be read",
    "NR 需要 64 位 Windows": "NR requires 64-bit Windows",
    "尚未确认 DXGI/CUDA 物理卡映射；先列出设备并选择": "No physical GPU is selected; click List GPUs, then select an adapter",
    " 不是有效的 64 位 Windows DLL": " is not a valid 64-bit Windows DLL",
    "NR 环境配置不可用；没有切换显卡或运行库": "NR environment settings are unavailable; no device or runtime was changed",
    "先显式列出物理显卡，再选择另一张卡": "Click List GPUs before choosing another physical adapter",
}


def forge_language(localization):
    locale = str(localization or "").strip().casefold().replace("_", "-")
    return "zh" if locale in ("zh", "chinese", "简体中文", "中文") or locale.startswith(("zh-cn", "zh-hans")) else "en"


def preparation_message(error, language="en"):
    code = getattr(error, "code", "")
    known = {"source_missing", "source_invalid", "preparing", "dependencies_missing", "dependencies_failed", "repair_blocked",
             "manual_missing", "manual_invalid", "mode_invalid"}
    if code in known:
        key = code
    elif isinstance(error, HTTPError) and error.code in (401, 403, 404):
        key = "download_unavailable"
    elif isinstance(error, (URLError, TimeoutError, ConnectionError)):
        key = "download_failed"
    elif code in ("incomplete", "limit", "archive_invalid", "size_invalid"):
        key = "download_invalid"
    elif isinstance(error, PermissionError):
        key = "storage_failed"
    else:
        key = "prepare_failed"
    return text(key, language)


def text(key, language="en", **values):
    if language not in CATALOGS:
        raise ValueError("Unsupported UI language")
    if language == "en" and "error" in values:
        error = str(values["error"])
        for original, translated in ERRORS_EN.items():
            error = error.replace(original, translated)
        values["error"] = error
    return CATALOGS[language][key].format(**values)