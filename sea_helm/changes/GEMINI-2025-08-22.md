# How to Install pyonmttok?

The repository SEA-HELM provides requirements.txt. And the following command

```shell
$ pip3 install -r requirements.txt
```

fails with the following error.

```shell
   ...
ERROR: Ignored the following versions that require a different python version: 0.1.0 Requires-Python <3.12,>=3.8
ERROR: Could not find a version that satisfies the requirement pyonmttok==1.37.1 (from versions: none)
ERROR: No matching distribution found for pyonmttok==1.37.1
$
```

The original requirements.txt failed with the following errors. 

| The original requirements.txt `bert-score==0.3.13 datasets==3.0.1 einops==0.8.0 evaluate==0.4.3 fast-langdetect==0.2.2 litellm==1.61.20 nltk==3.9.1 omegaconf==2.3.0 openai==1.64.0 pyonmttok==1.37.1 pythainlp==5.0.5 sacrebleu==2.4.3 scikit-learn==1.4.2 transformers==4.48.3`  Q: Downgrade the Python version? A: Doing so may cause additional issue with the recent versions of other tools like LiteLLM, OLLama, Gemini-CLI.  |
| :---- |

| Instead, try to use a new requirements.txt for the new versions. `# Core dependencies with flexible versioning for Python 3.13 compatibility bert-score datasets einops evaluate fast-langdetect litellm nltk omegaconf openai pythainlp sacrebleu scikit-learn transformers vllm openai-agents[litellm]  # Vertex AI dependencies for Gemini API google-cloud-aiplatform>=1.35.0 google-auth>=2.17.0 google-auth-oauthlib>=1.0.0 google-auth-httplib2>=0.1.0 vertexai>=1.35.0 google-genai>=0.3.0  # Optional dependencies (commented out if causing issues) # pyonmttok>=1.37.1  # May not be compatible with Python 3.13`  There is one problem left with the new requirements.txt. I wasn't able to install pyonmttok properly.  Note: The tokenizer package pyonmttok is missing. Manual installation attempts failed `$ sudo apt install -y build-essential cmake`  `$ pip install git+https://github.com/OpenNMT/Tokenizer.git`  `Collecting git+https://github.com/OpenNMT/Tokenizer.git   Cloning https://github.com/OpenNMT/Tokenizer.git to /tmp/pip-req-build-xnwtw34f   Running command git clone --filter=blob:none --quiet https://github.com/OpenNMT/Tokenizer.git /tmp/pip-req-build-xnwtw34f   Resolved https://github.com/OpenNMT/Tokenizer.git to commit 75117755d6c7a24d46509f867726e446be05317f   Running command git submodule update --init --recursive -q ERROR: git+https://github.com/OpenNMT/Tokenizer.git does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found. $`  What should I do to fix this problem and install pyonmttok successfully? |
| :---- |

## Gemini 2.5 Pro Deep Research
```
Investigate if a newer version of pyonmttok is available that supports Python 3.13.
```


Technical Report on the Python Version Compatibility of the pyonmttok Library
1.0 Executive Summary: Current Support Status for Python 3.13
An exhaustive investigation into the pyonmttok library concludes that no version currently supports Python 3.13. Furthermore, the analysis reveals that official support has not yet been extended to the preceding major version, Python 3.12. This lack of compatibility is not a temporary oversight but rather a systemic issue indicated by multiple, consistent sources of evidence.
The key findings of this report are as follows:
Explicit Dependency Constraints: The official package metadata published on the Python Package Index (PyPI) explicitly restricts the library's installation to Python versions less than 3.12. This constraint, defined as Requires: Python >=3.6, <3.12, is enforced by standard installation tools like pip.
Absence of Pre-Compiled Binaries: The latest available version of the library, pyonmttok 1.37.1, provides pre-compiled binary wheels for CPython up to version 3.11. There are no official binaries available for Python 3.12 or 3.13, making direct installation on these versions impossible without building from source.
Unresolved Python 3.12 Installation Failure: A critical GitHub issue, #329, was opened on May 31, 2024, detailing the failure of pyonmttok to install on a Python 3.12 environment. This issue remains open and unaddressed by the project maintainers, establishing a significant technical precedent that effectively blocks any potential support for Python 3.13 until resolved.
Slowing Development Cadence: The project's release history and recent development activity indicate a reactive maintenance posture. The last official release was on March 1, 2023, and subsequent activity has focused on minor dependency updates rather than major compatibility work. This cadence has fallen behind Python's annual release cycle.
The strategic implication for technical teams is clear: any project that requires Python 3.12 or a newer version cannot currently use pyonmttok. Stakeholders must either pin their project's environment to a compatible Python version, such as Python 3.11, or begin the process of evaluating alternative tokenization libraries that offer more active maintenance and broader version compatibility.
2.0 Analysis of Official Package Distribution and Stated Dependencies
The most definitive evidence regarding a Python package's compatibility is found in its official distribution channels and the metadata it declares. For pyonmttok, the information available on the Python Package Index (PyPI) provides a clear and unambiguous picture of its supported environments.
2.1 PyPI Metadata as the Official Contract
The pyonmttok project page on PyPI serves as the canonical source for its installation requirements. The metadata for the package explicitly defines the range of compatible Python versions using the classifier: Requires: Python >=3.6, <3.12. This declaration functions as a formal contract between the package maintainers and its users.
This metadata is not merely informational; it is a functional constraint that governs the behavior of package management tools. When a user executes a command such as pip install pyonmttok in a Python 3.12 or 3.13 environment, pip will read this constraint, determine that no compatible version of the package exists for the target environment, and fail the installation with an error. This is the primary and most direct piece of evidence confirming the lack of support for Python 3.12 and, by extension, any subsequent version, including 3.13.
2.2 Examination of Pre-compiled Binary Wheels
Further corroborating the metadata is the collection of files available for download on PyPI. pyonmttok is a library with C++ components, meaning it typically relies on pre-compiled binary files, known as "wheels," for straightforward installation across different platforms and Python versions. A detailed review of the files provided for the latest version, pyonmttok-1.37.1 (released March 1, 2023), reveals a distinct cutoff in Python version support.
The naming convention for wheel files explicitly denotes the targeted CPython version (e.g., cp311 for CPython 3.11). The available files for pyonmttok 1.37.1 cover versions from CPython 3.8 through 3.11. The complete absence of any files tagged with cp312 or cp313 provides concrete proof that no official binaries have been built or published for these newer Python versions. This means that even if the metadata constraint were not present, a standard pip install would fail on these versions because it would be unable to find a suitable wheel and would be forced to attempt a complex build from source.
The support matrix derived from the available files is summarized in the table below.
Python Version (CPython Tag)
Linux (manylinux_x86_64)
Linux (manylinux_aarch64)
macOS (x86_64)
macOS (arm64)
Windows (amd64)
Python 3.8 (cp38)
Available
Available
Available
Not Available
Available
Python 3.9 (cp39)
Available
Available
Available
Not Available
Available
Python 3.10 (cp310)
Available
Available
Available
Not Available
Available
Python 3.11 (cp311)
Available
Available
Available
Available
Not Available
Python 3.12 (cp312)
Not Available
Not Available
Not Available
Not Available
Not Available
Python 3.13 (cp313)
Not Available
Not Available
Not Available
Not Available
Not Available

Export to Sheets
Table 1: Python Wheel Availability for pyonmttok v1.37.1 on PyPI, based on data from.
2.3 Implications of Aligned Metadata and Wheel Availability
The perfect alignment between the PyPI metadata (<3.12) and the available binary wheels (up to 3.11) is highly significant. It demonstrates that the lack of Python 3.12 support is an intentional and known state of the project, not an accidental omission in the build or deployment process.
A scenario where a package supports a new Python version but fails to upload the corresponding wheel file would point to a simple logistical error. However, the maintainers of pyonmttok have taken the deliberate step of setting the metadata to actively prevent pip from even attempting an installation on Python 3.12. This action strongly implies that they are aware the package is incompatible and are proactively preventing users from encountering installation failures, runtime errors, or other undefined behavior. Therefore, the issue is not a superficial deployment problem but a more fundamental compatibility or maintenance challenge. This raises the critical question of why support ceased at Python 3.11, a question that community reports and the project's development history help to answer.
3.0 The Python 3.12 Precedent: A Critical Blocker for Future Versions
The lack of support for Python 3.12 is not just a standalone issue; it serves as a critical precedent and a technical barrier to any future compatibility with Python 3.13. The challenges preventing a Python 3.12 release must be addressed before work on any newer version can commence.
3.1 Analysis of GitHub Issue #329
On May 31, 2024, a community member opened issue #329 on the OpenNMT/Tokenizer GitHub repository, titled "pyonmttok installation fails while installing OpenNMT-py with python 3.12". This report provides a real-world validation of the conclusions drawn from the PyPI analysis.
The user details their attempt to install the package on Red Hat 9.4 with Python 3.12, which resulted in the error: error: Could not find suitable distribution for Requirement.parse('pyonmttok'). This is precisely the error message
pip produces when its dependency resolution process, guided by the package's metadata, fails to find a compatible version. Crucially, the user confirms that their "Installation was successful with python 3.11," effectively isolating the problem to the upgrade from Python 3.11 to 3.12. This community-provided evidence removes any ambiguity and confirms that the compatibility ceiling is firmly set at Python 3.11.
3.2 Project Response and Maintenance Velocity
The status of Issue #329 is as telling as its content. As of the latest available information, the issue remains open, has not been assigned to any developer, and has received no comments or acknowledgements from the project maintainers. Python 3.12 was officially released in October 2023, meaning this fundamental compatibility issue has been present for a considerable time and has gone unaddressed in the official repository for months since being reported.
This lack of engagement is a strong negative indicator of the project's current maintenance velocity and its capacity to respond to changes in the core Python ecosystem. For a library that is a key dependency in larger projects like OpenNMT-py, this delay in addressing a breaking compatibility issue represents a significant risk for its user base.
3.3 The Underlying Challenge of C++ Dependencies
The unresolved Python 3.12 issue is not merely a bug; it is a symptom of a deeper architectural reality. pyonmttok is the Python wrapper for OpenNMT/Tokenizer, a high-performance tokenization library written primarily in C++. The connection between the Python interface and the C++ backend is managed by a binding layer. The project's release notes indicate the use of
pybind11 for this purpose.
While this architecture provides significant performance benefits, it also introduces a substantial maintenance burden. The Python C-API, which C++ extensions rely on, is not guaranteed to be stable between major Python versions. Changes in the API can require non-trivial updates to the C++ binding code to ensure compatibility. The effort to update, recompile, and validate these C++ bindings for each new Python release is significantly higher than for a pure-Python package.
The barrier to Python 3.13 support is therefore not Python 3.13 itself, but the unaddressed technical work required for Python 3.12 compatibility. The project's maintenance cadence has fallen behind Python's predictable annual release schedule. The current situation suggests that the resources required to perform this C++ binding maintenance are either unavailable or have been deprioritized, creating a technical debt that blocks support for all future Python versions.
4.0 Project Development Trajectory and Release Cadence
A project's history of updates and its recent activity provide valuable context for its current state and future prospects. In the case of pyonmttok, the development trajectory shows a project that once actively kept pace with the Python ecosystem but has since slowed considerably.
4.1 Historical Python Version Adoption
The project's release notes on GitHub demonstrate a clear history of adopting new Python versions. Notably, the release of Tokenizer 1.34.0 included a specific new feature: "[Python] Build wheels for Python 3.11". This shows that the maintainers have an established process for extending compatibility, which involves build tools like
cibuildwheel, also mentioned in the release history.
This historical context makes the current situation more significant. The failure to support Python 3.12 is not due to a lack of process or precedent, but rather an apparent halt in the execution of that process. The machinery to support new Python versions exists within the project's development workflow, but it has evidently not been run for the Python 3.12 and 3.13 release cycles.
4.2 Analysis of Recent Development Activity
The last official release of pyonmttok on PyPI, version 1.37.1, was on March 1, 2023. This marks a substantial period of inactivity in terms of delivering new, stable versions to the user community. While development may occur between releases, the lack of a new version for over a year is a key indicator of slowed momentum.
An analysis of the OpenNMT/Tokenizer repository's recent activity corroborates this observation. Open pull requests primarily concern updates to dependencies, such as a fix for compiling with a newer version of the ICU library or bumps to GitHub Actions versions. There are no open pull requests specifically targeting Python 3.12 or 3.13 compatibility. Similarly, a review of open issues reveals the community-reported Python 3.12 problem but no proactive, maintainer-initiated issues or discussions related to supporting newer Python versions.
4.3 Shift to a Maintenance-Oriented Posture
Synthesizing these observations, it becomes clear that the project's development has transitioned from a phase of active feature development and proactive compatibility updates to a more passive "maintenance mode." Earlier release notes describe the addition of significant new features, such as the pyonmttok.Vocab utility. In contrast, the current activity is largely reactive, focused on fixing breakages caused by external ecosystem changes (e.g., an updated compiler or dependency) rather than advancing the library's own capabilities or compatibility footprint.
This shift has direct consequences for the project's ability to keep up with ecosystem evolution. Supporting a new major Python version is a significant development task, especially for a C++-backed library. It requires dedicated time for code updates, compilation across multiple platforms, testing, and release management. This type of proactive work appears to be outside the scope of the project's current resource allocation, which is focused on essential, break-fix maintenance.
5.0 Broader Compatibility Landscape and Support Posture
The challenges with Python version compatibility are not an isolated phenomenon but are part of a broader pattern of targeted, non-comprehensive platform support. This reveals a strategic allocation of limited resources that has important implications for users outside the project's core target environment.
5.1 Operating System Support Inconsistencies
The official documentation presents a conflicting picture of platform support. The README.md file in the GitHub repository lists Linux, macOS, and Windows as supported operating systems. However, this claim is directly contradicted by statements from project maintainers in community forums. In a 2021 discussion on the OpenNMT forum, a maintainer stated, "Sorry, we don't publish the pyonmttok package for Windows and there are currently no plans to do that".
The actual release files on PyPI provide a more nuanced, and still inconsistent, reality. As shown in Table 1, Windows wheels (win_amd64.whl) are available for Python 3.8, 3.9, and 3.10. However, a Windows wheel is notably absent for Python 3.11, even though Linux and macOS wheels for that version are provided. This demonstrates that Windows support is, at best, inconsistent, fragile, and a lower priority than support for Unix-like systems. The support that does exist already lags behind that of other platforms.
5.2 A Pattern of Targeted, Non-Comprehensive Support
This discrepancy between documented and actual support, combined with the lag in adopting new Python versions, reveals a clear pattern. The project's maintenance resources are evidently focused on ensuring functionality for a primary target environment, which is likely Linux and macOS, the dominant platforms in academic and research settings for machine learning. Other platforms and newer language versions, which fall outside this core focus, receive less attention and resources.
This support posture is a classic characteristic of a project that originated to solve a specific problem in a research context and has since gained a wider user base. Such projects are often highly optimized for their primary task—in this case, providing a fast and customizable tokenizer for the OpenNMT ecosystem —but may lack the dedicated, broad-based resources of commercially backed open-source libraries. Without such resources, it is difficult to handle the "long tail" of platform compatibility, comprehensive documentation, and timely community support across the entire software ecosystem. This context is critical for any team assessing the long-term viability and risks of adopting
pyonmttok as a dependency, particularly in a production environment that may have different platform or language version requirements than the project's original target audience.
6.0 Conclusion and Strategic Recommendations for Technical Stakeholders
The accumulated evidence from official package repositories, source code history, and community issue trackers leads to a clear and definitive conclusion regarding the compatibility of pyonmttok with modern Python versions. This conclusion, in turn, informs a set of strategic recommendations for technical teams and stakeholders who rely on this library.
6.1 Synthesis of Findings
The pyonmttok library is incompatible with Python 3.13 and its immediate predecessor, Python 3.12. This is not a speculative assessment but a conclusion based on a convergence of evidence:
Explicit Prohibition: The package's own metadata on PyPI explicitly forbids installation on Python versions $3.12$ and newer.
Missing Binaries: No official pre-compiled wheels are available for Python 3.12 or 3.13, making standard installation impossible.
Confirmed Failure: A public, unresolved GitHub issue confirms that installation fails on Python 3.12, establishing a known technical barrier.
Stalled Maintenance: The project's development cadence has slowed to a reactive, maintenance-oriented posture, and it has fallen behind the annual Python release cycle.
The root cause of this compatibility gap is not trivial. It is deeply tied to the architectural complexity of maintaining a C++-backed library with Python bindings across an evolving Python C-API. This task requires a significant and proactive investment of developer resources, which appear to be limited, as evidenced by the project's broader support posture and release inactivity.
6.2 Actionable Recommendations
Given this analysis, technical stakeholders should adopt a cautious and deliberate strategy regarding their use of pyonmttok. The following tiered recommendations are provided to guide decision-making based on project requirements and risk tolerance.
Immediate Strategy (Low Risk, High Compatibility): For any project currently using or planning to use pyonmttok, the most stable and recommended course of action is to pin the development and production environment to Python 3.11. This aligns with the latest officially supported version for which binary wheels are provided on all major platforms except Windows. This approach ensures predictable installation, avoids unresolved compatibility issues, and represents the lowest-risk path forward.
Proactive Monitoring (Low Effort, Future-Proofing): Teams should subscribe to notifications for GitHub Issue #329 on the OpenNMT/Tokenizer repository. This issue will serve as the primary and most direct channel for any official updates or community progress regarding Python 3.12 support. Any movement on this issue is a necessary prerequisite for any future work on Python 3.13. Monitoring this issue provides a low-effort way to stay informed of any changes to the library's compatibility status.
Strategic Alternative (Medium Effort, Risk Mitigation): If the use of Python 3.12 or a newer version is a non-negotiable project requirement—for instance, to leverage new language features or to maintain compatibility with other critical dependencies—teams should initiate a formal evaluation of alternative tokenization libraries. Libraries such as Hugging Face's tokenizers, which is backed by a large organization and has a strong track record of timely updates and broad platform support, may present a lower long-term maintenance risk.
Advanced Mitigation (High Risk, Last Resort): As a final option, and one that is not generally recommended, an advanced team with C++ expertise could attempt to build pyonmttok from source against Python 3.12 or 3.13. This would require setting up a C++ build environment with CMake, managing dependencies like ICU, and potentially patching the pybind11 C++ binding code to be compatible with the newer Python C-API. This path is entirely unsupported, carries a high risk of introducing subtle runtime bugs, and would create a private fork of the library that the team would be solely responsible for maintaining. This should only be considered as a temporary stopgap measure in a mission-critical situation where no other alternative is feasible.

