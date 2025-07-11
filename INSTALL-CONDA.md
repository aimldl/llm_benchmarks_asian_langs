# Installing Anaconda

This guide summarizes essential commands for installing Anaconda on Linux.

### 1. Install Anaconda on Linux

It's recommended to use Anaconda for managing your Python environments and packages.

#### Download the Anaconda Installer

First, download the latest Anaconda installer for Linux. Always download from the [official Anaconda website](https://www.anaconda.com/download/success) to ensure you get a secure and up-to-date version.

You can use `wget` to download the installer directly to your home directory. Open a terminal and run the following command. **Remember to replace the URL with the latest version available from the Anaconda repository.**

```bash
wget -P ~/ [https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh)
```
You can check for the latest version on the [Anaconda download success page](https://www.anaconda.com/download/success).

#### Verify the Installer's Integrity (Optional but Recommended)
To ensure the installer wasn't corrupted during download, you can verify its SHA-256 checksum. First,

```bash
sha256sum ~/Anaconda3-2025.06-0-Linux-x86_64.sh
```
Then, compare the output with the [official checksums](https://www.anaconda.com/download/success) provided on the Anaconda website.

#### Run the Installation Script
Once the download is complete (and verified), you can run the installation script using bash:

```bash
bash ~/Anaconda3-2025.06-0-Linux-x86_64.sh
```
During the installation process, you will be prompted to:

- Review and accept the license agreement: Press `Enter` to review the terms and then type `yes` to agree.
- Choose an installation location: The default location is your home directory `~/anaconda3`. It is generally recommended to accept the default unless you have a specific reason to change it.
- Initialize Anaconda: The installer will ask if you wish to initialize Anaconda3 by running `conda init`. It is recommended to select `yes`. This will modify your shell's startup script (e.g., `.bashrc`) to add the `conda` executable to your system's `PATH`.

#### Activate the Installation
After the installation is complete, you need to reload your shell's configuration file for the changes to take effect. You can do this by closing and reopening your terminal or by running:

```bash
source ~/.bashrc
```
For example,
```bash
$ source ~/.bashrc
(base) $
```
