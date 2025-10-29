#include <nvtx3/nvToolsExt.h>

void nvtxDomainCreate_c(const char* name, nvtxDomainHandle_t* domain)
{
    nvtxDomainHandle_t dom = nvtxDomainCreate(name);
    *domain = dom;
}

void nvtxDomainRangePushEx_c(nvtxDomainHandle_t domain, const char* message, const int color)
{
    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;

    nvtxDomainRangePushEx(domain, &eventAttrib);
}

void nvtxDomainRangePop_c(nvtxDomainHandle_t domain)
{
    nvtxDomainRangePop(domain);
}