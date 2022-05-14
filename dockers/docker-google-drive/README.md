# Docker google-drive-ocamlfuse
    Docker image to mount a google drive with google-drive-ocamlfuse shared with host.


### Build (local)
````
cd dockers/docker-google-driv
docker build -t   artia37/docker-google-drive:v1 -f Dockerfile .
````

### Usage
````
docker pull artia37/docker-google-drive:v1
docker run -d \
        -e CLIENT_ID='my-client-id' \
        -e CLIENT_SECRET='my-client-secret' \
        -e LAST_ACCESS_TOKEN='my-last-access-token' \
        -e REFRESH_TOKEN='my-refresh-token' \
        --security-opt apparmor:unconfined \
        --cap-add mknod \
        --cap-add sys_admin \
        --device=/dev/fuse \
        -v /mnt/drive:/mnt/gdrive:shared \
        artia37/docker-google-drive:v1
````

### Structure
* ls - '/mnt/gdrive'  #### Google Drive Fuse mount directory inside container








----------------------------------------------------------------------------------------------------

### Auto mount is here:

https://github.com/astrada/google-drive-ocamlfuse/issues/591



https://dev.to/yawaramin/use-google-drive-as-a-local-directory-on-linux-1b9





### Environment Variables
  * `PUID`: User ID to run google-drive-ocamlfuse
  * `PGID`: Group ID to run google-drive-ocamlfuse
  * `MOUNT_OPTS`: Additional mount options (user_allow_other is configured in /etc/fuse.conf)
  * `CLIENT_ID`: Google oAuth client ID without trailing `.apps.googleusercontent.com`
  * `CLIENT_SECRET`: Google oAuth client secret
  * `LAST_ACCESS_TOKEN`: Get value of `last_access_token` at ~/.gdfuse/default/state
  * `REFRESH_TOKEN`: Get value of `refresh_token` at ~/.gdfuse/default/state
  * `TEAM_DRIVE_ID`: (Optional) Team Drive Id to access a team folder instead of your private folder. The id can be found in the URL if you open the team folder in your browser (e.g. https://drive.google.com/drive/u/1/folders/${TEAM_DRIVE_ID})


### Host Configuration
1. If using systemd to manage the docker daemon process make sure that the service is configured either explicitly with a `shared` mountflag or un-configured and defaults to `shared`.
2. The mount point will also need to have it's propagation explicitly marked as shared.

Without this the fuse mount will not propagate back to the host.

````
# Ensure docker daemon uses shared mount flags
sed -i 's/MountFlags=\(private\|slave\)/MountFlags=shared/' /etc/systemd/system/docker.service
systemctl daemon-reload
systemctl restart docker.service
````

````
# Sample systemd for docker daemon with MountFlags
[Unit]
Description=Docker Application Container Engine
Documentation=https://docs.docker.com
After=network-online.target docker.socket firewalld.service containerd.service
Wants=network-online.target
Requires=docker.socket containerd.service

[Service]
Type=notify
# the default is not to use systemd for cgroups because the delegate issues still
# exists and systemd currently does not support the cgroup feature set required
# for containers run by docker
ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
ExecReload=/bin/kill -s HUP $MAINPID
TimeoutSec=0
RestartSec=2
Restart=always
MountFlags=shared

# Note that StartLimit* options were moved from "Service" to "Unit" in systemd 229.
# Both the old, and new location are accepted by systemd 229 and up, so using the old location
# to make them work for either version of systemd.
StartLimitBurst=3

# Note that StartLimitInterval was renamed to StartLimitIntervalSec in systemd 230.
# Both the old, and new name are accepted by systemd 230 and up, so using the old name to make
# this option work for either version of systemd.
StartLimitInterval=60s

# Having non-zero Limit*s causes performance problems due to accounting overhead
# in the kernel. We recommend using cgroups to do container-local accounting.
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity

# Comment TasksMax if your systemd version does not support it.
# Only systemd 226 and above support this option.
TasksMax=infinity

# set delegate yes so that systemd does not reset the cgroups of docker containers
Delegate=yes

# kill only the docker process, not all processes in the cgroup
KillMode=process
OOMScoreAdjust=-500

[Install]
WantedBy=multi-user.target
````

````
# Specify the mount points propagation as shared (execute as root)
mkdir -p /mnt/drive
mount --bind /mnt/drive /mnt/drive
mount --make-shared /mnt/drive
````

