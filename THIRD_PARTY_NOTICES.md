# Third-Party Notices

CausticaLM's project-owned code is licensed under `LGPL-3.0-or-later`. This file
documents third-party components and license boundaries that are not changed by
CausticaLM's license.

## NVIDIA DLSS / NGX SDK

CausticaLM can build and distribute release artifacts that include NVIDIA DLSS/NGX
SDK runtime components, including DLSS Ray Reconstruction and Frame Generation
libraries. These NVIDIA components are proprietary third-party software and are
not licensed under the LGPL.

The NVIDIA SDK components remain subject to the NVIDIA RTX SDKs license:

<https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt>

The LGPL license grant for CausticaLM does not grant rights to NVIDIA SDK
components. Redistribution and use of those components must comply with
NVIDIA's license terms.

This software contains source code provided by NVIDIA Corporation.

Bundled NVIDIA SDK runtime libraries may include files matching:

- `caustica/natives/windows-x64/nvngx_dlssd.dll`
- `caustica/natives/windows-x64/nvngx_dlssg.dll`
- `caustica/natives/linux-x64/libnvidia-ngx-dlssd.so*`
- `caustica/natives/linux-x64/libnvidia-ngx-dlssg.so*`

CausticaLM's `ngxshim` native library is project-owned glue code and follows
CausticaLM's project license unless otherwise noted.

## NVIDIA Real-time Denoisers (NRD)

CausticaLM can build and distribute release artifacts that include NVIDIA Real-time
Denoisers, linked statically into the `caustica_nrd` shim together with its
SPIR-V shader blobs. NRD is proprietary third-party software and is not licensed
under the LGPL.

NRD remains subject to the NVIDIA RTX SDKs license:

<https://github.com/NVIDIA-RTX/NRD/blob/main/LICENSE.txt>

The LGPL license grant for CausticaLM does not grant rights to NRD. Redistribution
and use of NRD must comply with NVIDIA's license terms.

This software contains source code provided by NVIDIA Corporation.

The NRD sources are never part of this repository. The shim's build fetches them
from NVIDIA's own distribution channel at configure time
(`https://github.com/NVIDIA-RTX/NRD.git`, pinned to a release tag), and only the
resulting object code is shipped. Bundled NRD runtime libraries may include files
matching:

- `caustica/natives/windows-x64/caustica_nrd.dll`
- `caustica/natives/linux-x64/libcaustica_nrd.so`

CausticaLM's `caustica_nrd` shim is project-owned glue code and follows CausticaLM's
project license unless otherwise noted. Its host-side shader ports
(`shaders/pipelines/nrd_frontend`, `shaders/pipelines/nrd_backend`) reimplement
encoding helpers described in NRD's `NRD.hlsli` and carry the same notice.
