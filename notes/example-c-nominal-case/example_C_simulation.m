clear;
s = sim('example_C_simulation_of_nominal_case');
assert(isequal(cont.Time, disc.Time))
t = table();
t.Time = cont.Time;
t.cont = cont.Data;
t.disc = disc.Data;
writetable(t, "example_C_simulation.csv");

