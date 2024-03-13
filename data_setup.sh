#!/bin/bash

# Alexa dataset download
wget 'http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip'

# OSINT (Bambenek consulting feeds) dataset download
wget 'https://faf.bambenekconsulting.com/feeds/dga-feed.txt' -O bambenek_dga_feed.txt

# Majestic Million
wget 'https://downloads.majestic.com/majestic_million.csv'