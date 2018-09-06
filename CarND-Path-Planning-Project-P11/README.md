# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program

### Introduction
In this project the goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. We will be provided with the car's localization and sensor fusion data, and also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

### Simulator

The Simulator which contains the Path Planning Project can be downloaded here :(https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).



## Reflection surrounding the problematic

This project is an excellent approach to self driving car path planning ,prediction , behavior planning and trajectory generation problematics. Indeed a fairly close/simple environment such as being on a highway  already represent multiple challenges to address! 

**Security first !**    

Collision avoidance has been addressed reading our car sensor fusion data , which we have been using for predicting the detected car trajectory and setting front and rear security distance with regards to the other car.

```c++
//define a secure distance to keep from front vehicle in same lane
int security_distance = 35; //(in frenet coordinate)
```

For the project implementation our pipeline was to first start with having our car staying in the same lane and to not collide with other vehicle ,  and then work on some lane changing scenario and smoothing the path still keeping some sort of distance security when going though the Lane change process.

```c++
//define boundaries for Lane change
double min_front_dist = 25;
double min_rear_distance = 10;
```

**Speed limit**

At any time we make sure not to break the speed limit and to also maintain an optimal speed on the highway overtaking vehicle in front of us that are to slow if needed.

```C++
//Speed limit on the highway
double speed_limit = 50.0;
```

**Comfort **

Since comfort is also an important point for our passenger , we have set several rules :

When possible the acceleration/braking always occur smoothly by slowly incrementing/decrementing the our vehicle speed. 

```C++
//Define a preferred changed rate in acceleration that don't generate discomfort
double acceleration_change_rate = 0.23;
```

We've also worked on smoothing our pass to no generate too much Jerk for the passenger, using spline trajectory generation. Another approach could also have been using Polynomials. 

**Path planning**

Our path is generated creating a spline going though 3 anchor points that are good enough spaced from one another as bellow :

```c++
 //Lets generate 3 anchor points each spaced by 30 (on s scale) to generate our spline

vector<double> anchorpoint0 = getXY(car_s + 30, 4*lane + 2, map_waypoints_s, map_waypoints_x, map_waypoints_y);

vector<double> anchorpoint1 = getXY(car_s + 60, 4*lane + 2, map_waypoints_s, map_waypoints_x, map_waypoints_y);

vector<double> anchropoint2 = getXY(car_s + 90, 4*lane + 2, map_waypoints_s, map_waypoints_x, map_waypoints_y);

```

We've bee using the Spline.h library to generate smooth trajectory to avoid jerk , remain in our lane if needed or change Lane according to our machine state logic. Based on our 3 anchor point we are generating our spline , also taking into account our previous path so that the new path continues it smoothly and don't generate any discomfort with  sudden changes with our previous path.

```c++
 //create a spline
            tk::spline s;
            //set our 3 spline anchor points
            s.set_points(ptsx, ptsy);

            // Set our horizon
            double target_x = 30;
            double target_y = s(30);
            double target_dist = sqrt(target_x*target_x + target_y*target_y);

            //Add  points from our previous path that have not been dealing with to our current path
            for (int i=0; i<prev_size; i++){
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
            }

            double x_add_on = 0.0;

            for (int i=0; i<50 - prev_size; i++){
              double N = target_dist/(0.02*ref_velocity/2.24);//dividing by 2.24 to convert miles per hours in meter per sec
              double x_ = x_add_on + target_x/N;
              double y_ = s(x_);
              x_add_on = x_;
```

We are also using our path to set our vehicle speed as not only we are telling our car where to  but when !

**Machine state logic and cost functions**

in order for our car to evolve safer on its lane and to perform a lane change we have been implementing 3 cost function that our car used to evaluate staying in the same lane, changing lane and even changing right or left when in the middle lane. The cost function are as follow :

```C++
//Define a boolean function calculating if we are beeing too close from a car or not.
bool too_close(double vehicle_s, double car_s, double upper_dist, double lower_dist){
  if (((vehicle_s - car_s) > upper_dist) or ((vehicle_s - car_s) < lower_dist)){
    return false;
  } else {
    return true;
  }
}

//define a cost function that calculate the cost of changing lane
double lane_change_cost(vector<double> lane_s, double car_s, double upper_dist, double lower_dist){
  double cost = 0;
  for(int i=0; i<lane_s.size(); i++){
    bool is_too_close = too_close(lane_s[i], car_s, upper_dist, lower_dist);
    if (is_too_close){
      cost += 10;
    }
    cost += 1; // number of vehicle in the lane
  }
  return cost;
}

//define a cost function that penalize staying in the same lane.
double keepLaneCost (double front_car, double car_d, double car_s, double lane) {
  double cost = 0;
  if ((car_d < (4*lane+2+2)) && (car_d > (4*lane))){
    if ((front_car > car_s) && ((front_car - car_s) < 30)){
      cost += 1;
    }
  }
  return cost;
}
```

When we are being stuck behind car in front of us that is to slow, our car will look at changing lane and will calculate the cost associated to it taking into account the space their is to change lane with regard to vehicle on this lane and also with regards to the number of vehicle on this lane. Surely we prefer changing for a clear land than for a busy one ! the logic can be found in our main.cpp file.



#### The map of the highway is in data/highway_map.txt

Each waypoint in the list contains  [x,y,s,dx,dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet s value, distance along the road, goes from 0 to 6945.554.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

## Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.

## Tips

A really helpful resource for doing this project and creating smooth trajectories was using http://kluge.in-chemnitz.de/opensource/spline/, the spline function is in a single hearder file is really easy to use.

---

## Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!


## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./



