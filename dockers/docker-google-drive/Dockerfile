FROM ubuntu:xenial
MAINTAINER utilmy

ENV DRIVE_PATH="/mnt/gdrive"

RUN echo "deb http://ppa.launchpad.net/alessandro-strada/ppa/ubuntu xenial main" >> /etc/apt/sources.list \
 && echo "deb-src http://ppa.launchpad.net/alessandro-strada/ppa/ubuntu xenial main" >> /etc/apt/sources.list \
 && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F639B041 \
 && apt-get update \
 && apt-get install -yy google-drive-ocamlfuse fuse \
 && apt-get clean all \
 && echo "user_allow_other" >> /etc/fuse.conf \
 && rm /var/log/apt/* /var/log/alternatives.log /var/log/bootstrap.log /var/log/dpkg.log


############################################################################################
#### Mount Google Drive
##### ENV Variables
# -e CLIENT_ID='my-client-id' \
# -e CLIENT_SECRET='my-client-secret' \
# -e LAST_ACCESS_TOKEN='my-last-access-token' \
# -e REFRESH_TOKEN='my-refresh-token' \

#### Addition
# --security-opt apparmor:unconfined \
# --cap-add mknod \
# --cap-add sys_admin \
# --device=/dev/fuse \
# -v /mnt/drive:/mnt/gdrive:shared \
# artia37/docker-google-drive:v1

### RUN apt-get install -yy google-drive-ocamlfuse fuse 
COPY mount_gdrive.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/mount_gdrive.sh
CMD ["/usr/local/bin/mount_gdrive.sh"]

# healthcheck
HEALTHCHECK cmd mount | grep -q "google-drive-ocamlfuse" || exit 1
############################################################################################




# Doc Steps
# 1) CLIENT_ID and CLIENT_SECRET are google oAuth client ID & client secret
#     Follow this guide to get those values
#     https://github.com/astrada/google-drive-ocamlfuse/wiki/Headless-Usage-&-Authorization
#
# 
# 2) Then run:    google-drive-ocamlfuse -id <CLIENT_ID> -secret <CLIENT_SECRET>
#     A browser will be started pointing to an authorization page, and when you allow access to Google Drive, 
#     a verification code will be generated, called :
# 
#
# 3) Copy this verification code and paste it in the console where the application is running.
#    application will notify you that it has retrieved the tokens from Google.
#   
#     LAST_ACCESS_TOKEN & REFRESH_TOKEN are generated automate whenever you re-run authorization
#     Those values are saved in ~/.gdfuse/default/state
#     Please copy manuall from there   ~/.gdfuse/default/state




