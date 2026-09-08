// SPDX-License-Identifier: MIT
// Copyright (c) 2026 TagSystemV2 contributors
// Host wrapper for lisitskyaa/ComfyUI-DLSS5-NR at exactly
// a3de4781eef81afe80d0226d1ede0b46b3346a63. See LICENSE and UPSTREAM.json.
// The upstream source is downloaded into ignored work/_gen, never edited.
#include "dlss5nr_bridge.cpp"

#include <sstream>

namespace tagsystem_nr {

static std::string Escape(const std::string& value) {
    std::string out = "\"";
    const char* digits = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 32) { out += "\\u00"; out += digits[c >> 4]; out += digits[c & 15]; }
        else out += static_cast<char>(c);
    }
    return out + "\"";
}

static std::string Utf8(const wchar_t* value) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (n < 1) return {};
    std::vector<char> buffer(static_cast<size_t>(n));
    WideCharToMultiByte(CP_UTF8, 0, value, -1, buffer.data(), n, nullptr, nullptr);
    return buffer.data();
}

// CUDA returns the eight native bytes of a Windows LUID. This representation
// intentionally does not reverse LowPart/HighPart or infer an ordinal from DXGI.
static std::string LuidBytes(const LUID& luid) {
    static_assert(sizeof(LUID) == 8, "Expected native Windows LUID layout");
    const auto* bytes = reinterpret_cast<const unsigned char*>(&luid);
    const char* digits = "0123456789abcdef";
    std::string out;
    for (int i = 0; i != 8; ++i) { out += digits[bytes[i] >> 4]; out += digits[bytes[i] & 15]; }
    return out;
}

static bool EnvironmentPresent(const wchar_t* name) {
    SetLastError(ERROR_SUCCESS);
    const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
    return size != 0 || GetLastError() != ERROR_ENVVAR_NOT_FOUND;
}

struct CudaAdapter { LUID luid{}; int ordinal = -1; };

static std::vector<CudaAdapter> CudaLuids(std::vector<std::string>& warnings) {
    std::vector<CudaAdapter> result;
    wchar_t order[128]{};
    GetEnvironmentVariableW(L"CUDA_DEVICE_ORDER", order, 128);
    if (EnvironmentPresent(L"CUDA_VISIBLE_DEVICES") ||
        (order[0] && wcscmp(order, L"FASTEST_FIRST") != 0)) {
        warnings.emplace_back("CUDA environment overridden: cuda:N is not a verified default physical device; launch a clean-environment inspection");
        return result;
    }
    wchar_t system[MAX_PATH]{};
    if (!GetSystemDirectoryW(system, MAX_PATH)) {
        warnings.emplace_back("Cannot locate the system NVIDIA CUDA driver; adapters remain unmapped");
        return result;
    }
    const std::wstring path = std::wstring(system) + L"\\nvcuda.dll";
    HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module) {
        warnings.emplace_back("System nvcuda.dll is unavailable; adapters remain unmapped");
        return result;
    }
    struct ModuleGuard { HMODULE value; ~ModuleGuard() { FreeLibrary(value); } } guard{module};
    // Minimum driver ABI declarations only. No NVIDIA SDK headers/runtime are
    // redistributed, and no cuCtxCreate/cuDevicePrimaryCtxRetain call exists.
    using Init = int(WINAPI*)(unsigned int);
    using Count = int(WINAPI*)(int*);
    using Device = int(WINAPI*)(int*, int);
    using GetLuid = int(WINAPI*)(char*, unsigned int*, int);
    const auto init = reinterpret_cast<Init>(GetProcAddress(module, "cuInit"));
    const auto count = reinterpret_cast<Count>(GetProcAddress(module, "cuDeviceGetCount"));
    const auto device = reinterpret_cast<Device>(GetProcAddress(module, "cuDeviceGet"));
    const auto get_luid = reinterpret_cast<GetLuid>(GetProcAddress(module, "cuDeviceGetLuid"));
    if (!init || !count || !device || !get_luid || init(0) != 0) {
        warnings.emplace_back("CUDA enumeration/LUID API unavailable; adapters remain unmapped");
        return result;
    }
    int total = 0;
    if (count(&total) != 0 || total < 0 || total > 1024) {
        warnings.emplace_back("cuDeviceGetCount failed; adapters remain unmapped");
        return result;
    }
    for (int i = 0; i < total; ++i) {
        int handle = -1;
        unsigned int mask = 0;
        CudaAdapter item;
        item.ordinal = i;
        if (device(&handle, i) == 0 &&
            get_luid(reinterpret_cast<char*>(&item.luid), &mask, handle) == 0 && mask != 0) {
            result.push_back(item);
        } else warnings.emplace_back("A CUDA device had no usable Windows LUID; no index-based fallback is permitted");
    }
    return result;
}

static std::string Devices() {
    std::vector<std::string> warnings;
    const auto cuda = CudaLuids(warnings);
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
        throw std::runtime_error("CreateDXGIFactory1 failed");
    std::ostringstream output;
    output << "{\"devices\":[";
    int nvidia_index = 0;
    for (UINT i = 0; ; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT status = factory->EnumAdapters1(i, &adapter);
        if (status == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(status)) throw std::runtime_error("DXGI adapter enumeration failed");
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) throw std::runtime_error("DXGI adapter description failed");
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) || desc.VendorId != 0x10de) continue;
        int matched = -1, matches = 0;
        for (const auto& item : cuda) {
            if (memcmp(&item.luid, &desc.AdapterLuid, 8) == 0) { matched = item.ordinal; ++matches; }
        }
        const std::string name = Utf8(desc.Description);
        std::string managed;
        if (matches == 1) managed = "cuda:" + std::to_string(matched);
        else warnings.emplace_back(name + ": LUID has no unique CUDA match; device is deliberately empty");
        if (nvidia_index) output << ',';
        // The upstream CreateDevice counts NVIDIA non-software adapters, not all
        // DXGI adapters. Keep that ABI index separate from CUDA's physical map.
        output << "{\"gpu_index\":" << nvidia_index++
               << ",\"gpu_name\":" << Escape(name)
               << ",\"device\":" << Escape(managed)
               << ",\"luid\":" << Escape(LuidBytes(desc.AdapterLuid))
               << ",\"total_memory\":" << static_cast<unsigned long long>(desc.DedicatedVideoMemory) << '}';
    }
    output << "],\"warnings\":[";
    for (size_t i = 0; i < warnings.size(); ++i) { if (i) output << ','; output << Escape(warnings[i]); }
    output << "],\"nr_initialization_performed\":false,\"cuda_context_created\":false}";
    return output.str();
}

} // namespace tagsystem_nr

extern "C" {

__declspec(dllexport) void __cdecl tagsystem_nr_reset_feature() {
    std::lock_guard<std::mutex> guard(g_mutex);
    // Same translation unit as the pinned upstream static release function.
    // Drop feature/frame/OFA history, keep the loaded DLL and D3D12 device.
    ReleaseFeatureAndResources();
    NvofReleaseSession();
}

__declspec(dllexport) const char* __cdecl tagsystem_nr_gpu_luid() {
    std::lock_guard<std::mutex> guard(g_mutex);
    static std::string value;
    value.clear();
    DXGI_ADAPTER_DESC1 desc{};
    if (g_initialized && g_adapter && SUCCEEDED(g_adapter->GetDesc1(&desc)))
        value = tagsystem_nr::LuidBytes(desc.AdapterLuid);
    return value.c_str();
}

__declspec(dllexport) const char* __cdecl tagsystem_nr_capabilities_json() {
    return "{\"host\":\"TagSystemV2\",\"abi\":1,\"hot_feature_reset\":true,"
           "\"cuda_luid_enumeration\":true,\"upstream\":\"a3de4781eef81afe80d0226d1ede0b46b3346a63\"}";
}

__declspec(dllexport) int __cdecl tagsystem_nr_devices_json(char* output, int capacity, char* error, int error_capacity) {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (error && error_capacity > 0) error[0] = '\0';
    try {
        const std::string result = tagsystem_nr::Devices();
        if (!output || capacity <= 0 || result.size() >= static_cast<size_t>(capacity))
            throw std::runtime_error("Device JSON output buffer is too small");
        memcpy(output, result.c_str(), result.size() + 1);
        return 1;
    } catch (const std::exception& exception) {
        if (error && error_capacity > 0) {
            const size_t n = std::min(strlen(exception.what()), static_cast<size_t>(error_capacity - 1));
            memcpy(error, exception.what(), n); error[n] = '\0';
        }
        if (output && capacity > 0) output[0] = '\0';
        return 0;
    }
}

} // extern C