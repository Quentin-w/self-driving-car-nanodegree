# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

In this project we will revisit the lake race track from the Behavioral Cloning Project. This time, however, we will implement a PID controller in C++ to maneuver the vehicle around the track!

The simulator will provide us with the cross track error (CTE) and the velocity (mph) in order to compute the appropriate steering angle.

------

In order to successfully drive across the track , we have been implementing a PID controller that drive the car steering angle. 

The overall PID formula can be found below as :

![{\displaystyle u(t)=K_{\text{p}}e(t)+K_{\text{i}}\int _{0}^{t}e(t')\,dt'+K_{\text{d}}{\frac {de(t)}{dt}},}](https://wikimedia.org/api/rest_v1/media/math/render/svg/69072d4013ea8f14ab59a8283ef216fb958870b2) 

Where : Kp, Ki , Kd are the PID Coefficient and E(t) our Cross track error.

In order to drive the car successfully across the track , the PID parameters needs to be optimal as each of them have an influence on the steering such as :



**Kp and the P-controller** 

The Kp term in the PID equation give an output directly proportional to the cross track error, this as directly an effect on  the car steering and is the most noticeable in the steering behavior ,  driving to the right will cause the P controller to steer the car proportionally in the opposite direction. A single P controller isn't a good design for steering and would makes most of the passenger sick! Not to mention a single P controller never reach a steady state condition.



**Kd anf the D-Controller**

The D controller allow us to anticipating future behavior of the error, as Its output depends on rate of change of error with respect to time it enable us to smooth the car steering and steer the car close to the road center, but not exactly! 



**Ki and the I-Controller**

Due to limitation of the p-controller and the D controller we need to introduce another term, the I controller and its associated coefficient Ki. The I controller has an integral term in its formulation that allow us to integrate the error over a period of time, the I controller , smooth or reduce the response of the PID controller but enable removing any bias that was preventing us reaching the road center position.



Combining these controller with the right parameters (Kp,Ki,Kd) enables us to get the desired response for our car steering system.



**Tuning our PID Controller**

Several method can be tried to get good PID parameters, trial and error is one of these and enable to well understand the impact each of the coefficient as on our system but can also be time consuming. More robust trail and error method and rules of thumbs exist to make it more effective. 

In this project we've tuned the PID manually, however other PID tuning technic such as gradient descent optimization, twiddle algorithm and more can assist finding the right parameters.

In our project the PID parameter that have been used are :

```c++
double init_Kp = 0.13;
double init_Ki = 0.0002;
double init_Kd = 3.0;

//Initialize our steer pid
steer_pid.Init(init_Kp, init_Ki, init_Kd);
```



Having tuned our PID the car successfully drive on the track !

Further work could be to design another PID for controlling the car throttle command and implement an optimization algorithm to further tune the PID parameters.



## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

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

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/e8235395-22dd-4b87-88e0-d108c5e5bbf4/concepts/6a4d8d42-6a04-4aa6-b284-1697c0fd6562)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
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

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

