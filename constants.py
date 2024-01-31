"""Module to hold constants shared between modules that would be difficult to properly load from configuration."""
import os

max_process_cnt = os.cpu_count()
max_process_cnt = max_process_cnt if max_process_cnt is not None else 4
max_process_cnt -= 2

module_to_attrs = {
    "http_ua": ["operating_system_ua", "operating_platform_ua", "hardware_type_ua"],
    "os_by_tcpip": ["os_by_tcpip"],
    "os_by_tls": ["tls_os_family", "tls_os_name", "tls_os_version", "tls_categories"],
    "sdp_labels": [
        "sdp_label",
        "dnssd_service",
        "dnssd_query",
        "ssdp_query",
        "ssdp_service",
    ],
    "tags_by_services": [
        "tags_by_services",
        "open_ports",
    ],
}
attr_to_module = {attr: module for module, attrs in module_to_attrs.items() for attr in attrs}

generic_labels = [
    "OperatingSystem.Windows",
    "OperatingSystem.Linux",
    "OperatingSystem.MacOS",
    "OperatingSystem.iOS",
    "OperatingSystem.Android",
]

non_reference_attr_columns = [
    "os_by_tcpip",
    "tags_by_services",
    "open_ports",
    "tls_os_family",
    "tls_os_name",
    "tls_os_version",
    "tls_categories",
    "sdp_label",
    "dnssd_service",
    "dnssd_query",
    "ssdp_query",
    "ssdp_service",
]

attr_columns = [
    "operating_system_ua",
    "hardware_type_ua",
    "operating_platform_ua",
    *non_reference_attr_columns,
]
