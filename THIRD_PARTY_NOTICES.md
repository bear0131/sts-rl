# Third-party components

本项目内含以下第三方组件（vendored，原样保留各自 LICENSE）。

## sts_lightspeed

- 路径：`sts-rl/cpp/sts_lightspeed/`
- 来源：https://github.com/gamerpuppy/sts_lightspeed
- 用途：RNG 精确的 Slay the Spire C++ 模拟器，作为模拟器内核。
- 许可：MIT License，见下文及 `sts-rl/cpp/sts_lightspeed/LICENSE.md`。
- 其自身又含两个 vendored 依赖：`json/`（nlohmann/json，MIT）、`pybind11/`（pybind11，BSD-3-Clause），许可见各自目录。

```
MIT License

Copyright (c) 2021 gamerpuppy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
