#!/bin/bash
# Keep the phone acting as the Jetson's network adapter over the USB cable.
#
# Why this exists: in the car the Jetson has no wifi it can reach. Its own radio
# cannot even hear the personal hotspot -- measured, with the hotspot's BSSID absent
# from a full scan while the Moto, cabled to the Jetson and so inches away, sat on it
# at RSSI -36. USB tethering sidesteps the air entirely: the phone presents itself as
# an ethernet adapter over the cable that is already there, the Jetson DHCPs on it,
# and Tailscale comes up over that.
#
# Why it is a loop rather than a one-shot: `svc usb setFunctions rndis` does NOT
# persist. `setScreenUnlockedFunctions rndis` was tried and did not take on this
# build -- `persist.sys.usb.config` stayed `adb` -- so the setting is lost on every
# replug and every phone reboot. Something has to put it back, and the Jetson is the
# only party that can, because it is the adb host.
#
# Setting rndis keeps adb: the resulting config is `rndis,none,adb`, verified on this
# handset. If it ever did drop adb this loop could not repair it, which is why the
# config is checked rather than set blindly -- a blind re-set every 20 s would bounce
# the USB connection continuously and take the drive's own link down with it.
set -u

INTERVAL="${DSRC_USB_NET_INTERVAL:-20}"

while true; do
    # Blocks until a handset is present, so a Jetson booted before the phone is
    # plugged in waits quietly instead of spinning.
    adb wait-for-device

    config=$(adb shell getprop sys.usb.config 2>/dev/null | tr -d '\r')
    case "$config" in
        *rndis*)
            : # already tethering; touching it would bounce the link for nothing
            ;;
        "")
            echo "$(date -Is) phone present but sys.usb.config unreadable; leaving it alone"
            ;;
        *)
            echo "$(date -Is) config is '$config'; enabling rndis"
            adb shell svc usb setFunctions rndis
            # The USB bus re-enumerates, which drops adb briefly and takes every
            # `adb reverse` mapping with it. run_demo re-establishes its own on the
            # next start; nothing here should try to.
            sleep 10
            echo "$(date -Is) config now: $(adb shell getprop sys.usb.config 2>/dev/null | tr -d '\r')"
            ;;
    esac

    sleep "$INTERVAL"
done
