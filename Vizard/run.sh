#!/bin/bash
if [ -z "$1" ];
    then echo "Must supply path to a directory where vizard_logger is saving its information";
    exit 0
fi
bokeh serve . --allow-websocket-origin=10.19.55.25:5006 --args $1 
