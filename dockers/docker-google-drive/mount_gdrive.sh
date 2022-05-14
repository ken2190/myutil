#!/bin/sh

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



##### CHECK IS ENV Variables exists  ###############################
# -e CLIENT_ID='my-client-id' \
# -e CLIENT_SECRET='my-client-secret' \
# -e LAST_ACCESS_TOKEN='my-last-access-token' \
# -e REFRESH_TOKEN='my-refresh-token' \



#####################################################################
DRIVE_PATH=${DRIVE_PATH:-/mnt/gdrive}

PUID=${PUID:-0}
PGID=${PGID:-0}

# Create a group for our gid if required
if [ -z "$(getent group gdfuser)" ]; then
	echo "creating gdfuser group for gid ${PGID}"
	groupadd --gid ${PGID} --non-unique gdfuser >/dev/null 2>&1
fi

# Create a user for our uid if required
if [ -z "$(getent passwd gdfuser)" ]; then
	echo "creating gdfuser group for uid ${PUID}"
	useradd --gid ${PGID} --non-unique --comment "Google Drive Fuser" \
	 --home-dir "/config" --create-home \
	 --uid ${PUID} gdfuser >/dev/null 2>&1

	echo "taking ownership of /config for gdfuser"
	chown ${PUID}:${PGID} /config
fi

# check if our config exists already
if [ -e "/config/.gdfuse/default/config" ]; then
	echo "existing google-drive-ocamlfuse config found"
else
	if [ -z "${CLIENT_ID}" ]; then
		echo "no CLIENT_ID found -> EXIT"
		exit 1
	elif [ -z "${CLIENT_SECRET}" ]; then
		echo "no CLIENT_SECRET found -> EXIT"
		exit 1
	elif [ -z "$LAST_ACCESS_TOKEN" ]; then
		echo "no LAST_ACCESS_TOKEN found -> EXIT"
		exit 1
	elif [ -z "$REFRESH_TOKEN" ]; then
		echo "no REFRESH_TOKEN found -> EXIT"
		exit 1		
	else
		echo "initializing google-drive-ocamlfuse..."
		mkdir -p /config/.gdfuse/default
		echo "last_access_token=$LAST_ACCESS_TOKEN" > /config/.gdfuse/default/state
		echo "refresh_token=$REFRESH_TOKEN" > /config/.gdfuse/default/state
		su gdfuser -l -c "google-drive-ocamlfuse -id $CLIENT_ID -secret ${CLIENT_SECRET}"

		# set teamdrive config"
		if [ -n "${TEAM_DRIVE_ID}" ];then
			sed -i "s/team_drive_id=/team_drive_id=${TEAM_DRIVE_ID}/g" /config/.gdfuse/default/config
		fi
	fi
fi

# prepend additional mount options with a comma
if [ -n "${MOUNT_OPTS}" ]; then
	MOUNT_OPTS=",${MOUNT_OPTS}"
fi

# mount as the gdfuser user
echo "mounting at ${DRIVE_PATH}"
exec su gdfuser -l -c "google-drive-ocamlfuse \"${DRIVE_PATH}\"\
 -f -o uid=${PUID},gid=${PGID}${MOUNT_OPTS}"



