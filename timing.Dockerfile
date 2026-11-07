FROM office-pr:smac-grid-45c479d1
ARG NATIVE_REV
COPY nav2_smac_planner/src/a_star.cpp /build_ws/src/nav2_smac_planner/src/a_star.cpp
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && source /workspace/install/setup.bash && cd /build_ws && MAKEFLAGS=-j2 colcon build --install-base /workspace/install --build-base /build_ws/build --executor sequential --packages-select nav2_smac_planner --allow-overriding nav2_smac_planner --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON"
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && source /workspace/install/setup.bash && ROS_DOMAIN_ID=179 ctest --test-dir /build_ws/build/nav2_smac_planner -R '^test_' --output-on-failure"
RUN NATIVE_REV=$NATIVE_REV python3 -c 'import json,pathlib,os; p=pathlib.Path("/workspace/nav2_source.json"); d=json.loads(p.read_text()); d["revision"]=os.environ["NATIVE_REV"]; p.write_text(json.dumps(d,indent=2))'
