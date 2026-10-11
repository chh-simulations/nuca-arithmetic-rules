% hca_canonical.m - MATLAB/Octave port of the canonical NuCA simulator.
% Mirrors the gated Python reference (python/hca_canonical.py) line for
% line. Self-test at the bottom carries checksums generated from that
% reference (deterministic initialization, no RNG): run this file; it
% errors if the port disagrees with the reference.
%
% State: N cells on a ring; c(j) in {0,1}; rule word w(j) = 8 bits.
% Reading e in {lsb, msb} fixes the integer value of a word. One step:
%   h(j) = c(j-1)+c(j)+c(j+1)           (periodic)
%   c(j) <- bit of w(j) at POSITION h(j) (position, not significance)
%   n(j) <- bitand(mod(n(j) + sgn*g(h) - 1, N), 255), sgn = 2c(j)-1
% Linear variant: g(h) = h.

function hca_canonical()
    checks = [ run_sim(64, 50, true);  run_sim(64, 50, false) ];
    expect = [ 0, 1785;  44, 2104 ];   % from the Python reference
    assert(isequal(checks, expect), 'port disagrees with reference');
    disp('hca_canonical.m: self-test passed (matches Python reference).');
end

function out = run_sim(N, steps, msb)
    words = mod(0:N-1, 256);           % deterministic init
    c = mod(0:N-1, 2);
    for s = 1:steps
        h = c + circshift(c, 1) + circshift(c, -1);
        for j = 1:N
            % bit at POSITION h(j): position p = bit index from the
            % word's own storage order; stored as integer n with the
            % chosen reading, position p corresponds to significance
            % p (lsb reading) or 7-p (msb reading).
            if msb, shift = 7 - h(j); else, shift = h(j); end
            c(j) = bitand(bitshift(words(j), -shift), 1);
            sgn = 2*c(j) - 1;
            words(j) = bitand(mod(words(j) + sgn*h(j) - 1, N), 255);
        end
    end
    out = [sum(c), sum(words)];
end
