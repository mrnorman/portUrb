#!/bin/bash

nodes=16  && grep "\<$1\>" frontier/nodes_$nodes/portUrb.out | awk 'BEGIN{t=0;c=0;mn=1e6;mx=0}{t+=$3;c++;mn=$3<mn?$3:mn;mx=$3>mx?$3:mx}END{print t/c "\n" mn "\n" mx}'
nodes=32  && grep "\<$1\>" frontier/nodes_$nodes/portUrb.out | awk 'BEGIN{t=0;c=0;mn=1e6;mx=0}{t+=$3;c++;mn=$3<mn?$3:mn;mx=$3>mx?$3:mx}END{print t/c "\n" mn "\n" mx}'
nodes=64  && grep "\<$1\>" frontier/nodes_$nodes/portUrb.out | awk 'BEGIN{t=0;c=0;mn=1e6;mx=0}{t+=$3;c++;mn=$3<mn?$3:mn;mx=$3>mx?$3:mx}END{print t/c "\n" mn "\n" mx}'
nodes=128 && grep "\<$1\>" frontier/nodes_$nodes/portUrb.out | awk 'BEGIN{t=0;c=0;mn=1e6;mx=0}{t+=$3;c++;mn=$3<mn?$3:mn;mx=$3>mx?$3:mx}END{print t/c "\n" mn "\n" mx}'
nodes=256 && grep "\<$1\>" frontier/nodes_$nodes/portUrb.out | awk 'BEGIN{t=0;c=0;mn=1e6;mx=0}{t+=$3;c++;mn=$3<mn?$3:mn;mx=$3>mx?$3:mx}END{print t/c "\n" mn "\n" mx}'
nodes=512 && grep "\<$1\>" frontier/nodes_$nodes/portUrb.out | awk 'BEGIN{t=0;c=0;mn=1e6;mx=0}{t+=$3;c++;mn=$3<mn?$3:mn;mx=$3>mx?$3:mx}END{print t/c "\n" mn "\n" mx}'

