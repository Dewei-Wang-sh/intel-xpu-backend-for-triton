#include "Driver/GPU/HsaApi.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace hsa {

struct ExternLibHsa : public ExternLibBase {
  using RetType = hsa_status_t;
#ifdef WIN32
  static constexpr const char *name = "hsa-runtime64.dll";
#else
  static constexpr const char *name = "libhsa-runtime64.so";
#endif
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = HSA_STATUS_SUCCESS;
  static void *lib;
};

void *ExternLibHsa::lib = nullptr;

DEFINE_DISPATCH(ExternLibHsa, agentGetInfo, hsa_agent_get_info, hsa_agent_t,
                hsa_agent_info_t, void *);

hsa_status_t iterateAgents(hsa_status_t (*callback)(hsa_agent_t agent,
                                                    void *data),
                           void *data) {
  typedef hsa_status_t (*hsa_iterate_agents_t)(
      hsa_status_t (*)(hsa_agent_t, void *), void *data);
  static hsa_iterate_agents_t func = nullptr;
  Dispatch<ExternLibHsa>::init(ExternLibHsa::name, &ExternLibHsa::lib);
  if (func == nullptr)
#ifdef WIN32
    func = reinterpret_cast<hsa_iterate_agents_t>(
        GetProcAddress((HMODULE)ExternLibHsa::lib, "hsa_iterate_agents"));
#else
    func = reinterpret_cast<hsa_iterate_agents_t>(
        dlsym(ExternLibHsa::lib, "hsa_iterate_agents"));
#endif
  return (func ? func(callback, data) : HSA_STATUS_ERROR_FATAL);
}

} // namespace hsa

} // namespace proton
