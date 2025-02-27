---
title: Setting Up RAID in Linux
date: 2025-02-18T21:34:36+08:00
type: post
categories: ["Introduction"]
tags: ["Tutorial", "Server", "Disk"]
image: disk-usage.png
# slug: RAID
# https://www.flaticon.com/
---
<a href="https://www.flaticon.com/free-icons/www" title="www icons">Www icons created by Prosymbols - Flaticon</a>

Disks are good. SSDs are better. But sometimes even SSDs are a bit of slow for your application.

In my case, the server has some SATA slots for 2.5 inch disks. The lab purchased some SATA SSDs. SATA SSDs have somewhat limited R/W bandwidth of about 550 MB/s. Sometimes you would want to load or write dataset files (like a CFD solution file) of multiple GBs, and repeatedly. Moreover, with multiple disks, each mounted disk is a separate file system and you would not like accessing them separately often times.

Of course, you can use logical volume manager to merge the disks, but RAID has the potential to increase bandwidth and data redundancy while merging the disks.

Here I will record my effort in building a RAID system in the lab.

## RAID

RAID stands for Redundant Array of Independent Disks, a technology designed to enhance performance and redundancy in data storage systems. By distributing data across multiple disks, RAID offers solutions tailored to different needs, whether it's speed, reliability, or a combination of both.

The primary types of RAID include:

- **RAID 0**: Known for striping data without parity, this configuration boosts performance by spreading data across disks but lacks redundancy.
- **RAID 1**: Mirrors data between two disks to provide redundancy, ensuring data availability if one disk fails.
- **RAID 5**: Uses distributed parity, offering a balance of performance and fault tolerance by allowing recovery from a single drive failure.
- **RAID 6**: Similar to RAID 5 but with double parity, providing enhanced redundancy against two drive failures at the cost of additional overhead.
- **RAID 10**: Combines mirroring and striping (RAID 1+0), offering both performance improvements and redundancy.

Each level serves specific purposes, making RAID a critical component in environments where data reliability, speed, and scalability are paramount, such as servers and enterprise systems.

| **RAID Level** | **Minimum Disks Required** | **Redundancy & Capacity**         | **Performance (Read/Write)**           |
| -------------- | -------------------------- | --------------------------------- | -------------------------------------- |
| **RAID 0**     | 2                          | No redundancy; $1/1$              | High read/write speeds                 |
| **RAID 1**     | 2                          | Data mirrored; $1/2$              | Fast reads, slower writes              |
| **RAID 5**     | 3                          | Single parity; $\frac{n-1}{n}$    | Balanced performance and redundancy    |
| **RAID 6**     | 4                          | Double parity; $\frac{n-2}{n}$    | Higher redundancy at the cost of space |
| **RAID 10**    | 4                          | Data mirrored and striped;  $1/2$ | Good read/write speeds with redundancy |

## Setting up a RAID5

### **Step 1: Install mdadm (if not already installed)**
The `mdadm` utility is used to manage RAID arrays on Linux systems.

```bash
sudo apt update && sudo apt install mdadm
```

---

### **Step 2: Verify Disk Availability and Prepare Disks**
Before setting up the RAID, ensure that all disks are available and not mounted or in use. You can list your disks with:

```bash
lsblk
# or
fdisk -l
```

If any of the disks contain data, back it up before proceeding, as this process will erase all data on the disks.

---

### **Step 3: Create the RAID 5 Array**
Use `mdadm` to create a new RAID 5 array. Replace `/dev/md0` with your desired array name (e.g., `/dev/md0`).

```bash
sudo mdadm --create --verbose /dev/md0 --level=5 --raid-devices=4 /dev/sda /dev/sdb /dev/sdc /dev/sdd
```

- `--create`: Specifies that you're creating a new RAID array.
- `--verbose`: Provides detailed output during the creation process.
- `--level=5`: Sets the RAID level to 5.
- `--raid-devices=4`: Indicates that you're using four devices for this array.

---

### **Step 4: Verify the RAID Creation**
Once the command completes, check the status of your new RAID array:

```bash
sudo mdadm --detail /dev/md0
```

This will display detailed information about the array, including its state, chunk size, and member disks.

You can also monitor the progress of the initial parity construction (resync) with:

```bash
cat /proc/mdstat
```

---

### **Step 5: Create a Filesystem on the RAID Array**
Once the RAID array is created, you need to create a filesystem on it. Here’s how to format it with `ext4`:

```bash
sudo mkfs.ext4 -v /dev/md0
```

- Replace `-v` with other options if needed (e.g., `-m 1` to set the reserved space for root).

---

### **Step 6: Mount the RAID Array**
Create a mount point and mount the new RAID array:

```bash
sudo mkdir /mnt/ssd-SATARAID5
sudo mount /dev/md0 /mnt/ssd-SATARAID5
```

You can now use `/mnt/ssd-SATARAID5` to store files.

If you want the array to mount automatically at boot, add an entry to `/etc/fstab`. First, find the UUID of your RAID array:

```bash
sudo blkid
```

with a return containing:

```bash
/dev/md0: UUID="XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" BLOCK_SIZE="4096" TYPE="ext4"
```

Then edit `/etc/fstab` and add a line like this:

`/etc/fstab`

```bash
UUID=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX /mnt/raid5 ext4 defaults 0 2
```

Replace `XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX` with the actual UUID of your array.

Note that although you can replace `UUID=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX` with `/dev/md0` just like in the `mount` command (or with `/dev/sdf1` if you are mounting a regular partition of some disk) and `mount -a` would actually work, rebooting the system or changing disk layouts could cause a mounting error during system boot. That's because the device and partition names like `/dev/sda` or `/dev/sda1` are not bonded to the physical disk or internal partition, while the UUIDs are safe to distinguish these devices.

---

### **Step 7: Save the RAID Configuration**
To ensure the RAID array is recognized after a reboot, update the initramfs:

```bash
sudo update-initramfs -u
```

---

### **Important Notes**

1. **Data Loss Warning**: This process will erase all data on the disks. Make sure you have backups of any important files before proceeding.

2. **RAID 5 Capacity**: With four disks, RAID 5 will provide you with usable space equivalent to three full disks (since one disk is used for parity).

3. **Monitoring the Array**: You can monitor the array’s status at any time with:

   ```bash
   sudo mdadm --detail /dev/md0
   ```

4. **Rebuilding the Array**: If a disk fails, you can replace it and rebuild the array using `mdadm`.

5. **Testing Failures**: To test redundancy, you can simulate a disk failure by removing one disk from the array (e.g., `sudo mdadm /dev/md0 --fail /dev/sda`), then adding it back.


## IO Benchmarking

Before benchmarking, wait until the parity building is done.

```bash
$ cat /proc/mdstat 
Personalities : [raid6] [raid5] [raid4] [raid0] [raid1] [raid10]
md0 : active raid5 sde[4] sdb[0] sdc[1] sdd[2]
      11720658432 blocks super 1.2 level 5, 512k chunk, algorithm 2 [4/4] [UUUU]
      bitmap: 0/30 pages [0KB], 65536KB chunk

unused devices: <none>
```

Sequential R/W:

```bash
#write
dd if=/dev/zero of=ssd1/test.test bs=1024M count=10 oflag=direct
#read
dd if=ssd1/test.test of=/dev/null bs=1024M  iflag=direct 
```

Here `ssd1` is the path to a place inside that disk you need to test.

4K R/W

```bash
#write
dd if=/dev/zero of=ssd1/test.test4k bs=4K count=65536 oflag=direct
#read
dd if=ssd1/test.test4k of=/dev/null bs=4K  iflag=direct 
```

Results:

| **Device**             | **Sequential Read** | **Sequential Write** | **4K Read**     | **4K Write**      |
| ---------------------- | ------------------- | -------------------- | --------------- | ----------------- |
| Single SATA SSD        | 567 MB/s            | 490 MB/s             | $141\pm14$ MB/s | $116\pm6$ MB/s    |
| RAID5 with 4 SATA SSDs | 2.0 GB/s            | 672 MB/s             | $117\pm12$ MB/s | $16.6\pm1.8$ MB/s |
| Single NVMe SSD        | 3.2 GB/s            | 798 MB/s             | $271\pm34$ MB/s | $217\pm18$ MB/s   |


It can be seen that 4K write is pretty poor compared to single SSDs. Maybe the computing overhead of software RAID affects much. 

4K read is also not good.

Sequential R/W improves much, but still not surpassing a single NVMe disk.

Maybe the correct usage of RAID is using hardware RAID solutions with commercial level devices. 

Anyway, in my case, consumer level SATA SSDs are much cheaper (in the sense of volume and the cost of slots on motherboard) than NVMe SSDs with PCI-E support, and software RAID `mdadm` provides a merged disk with parity and bandwidth improvement.
