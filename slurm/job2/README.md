# Run for Berkay

Run to test memory usage.

manual_sub.sh ---
https://github.com/beyucel/Pace_visualization/blob/main/job10_memorystudy/manual_sub.sh

The only difference here is I have 16 workers. The minimum memory for
this one was 33Gb. I want to check if we can go lower than 20GB.  I
think the number of chunks goes up to 72 for the manual_sub.sh. Can
you make it 1 20 40 80 120 160 200 240 280 320 chunks?

I did those studies for 8000 data points, I will also do it for 2000,
4000, and 6000 samples to see how it affects the values.  If you can
also do it for 2000, 4000, 6000, that would be great. ( you will only
need to update the manual_sub.sh file samples for-loop).


