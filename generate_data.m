% Generate data for impulse response estimation
% Niklas Wahlström 2017-05-17

n = 30; % The order of the systems
N = 125; % Number of time steps in each data set
Ng = 50; % Number of time steps for the impulse response to be stored
M = 1000000; % Number of data sets
%SNR = 10; % The signal to noise ratio in the data
% Using random SNR


plot_data = 0;
save_data = 1;

close all

% Allocate memory
U = zeros(M,N);
Y = zeros(M,N);
G = zeros(M,Ng);
SNR = zeros(M,1);

for i = 1:M
    if (mod(i,M/100) == 0 )
        i
    end
    
    bw = Inf;
    while(isinf(bw)) % Only consider systems with finite bandwidth
        m = rss(n); % Generate random system
        bw = bandwidth(m); % Compute the bandwidth
    end
    m.D = 0; % Set the directterm to zero
    f = 3*(bw*2*pi); %Compute the sampling frequency (3 times the bandwdidth)
    md = c2d(m,1/f,'zoh'); % Sample the systm
    md = md/dcgain(md); % Normalize the dcgain   
    md.Ts=1; % Set the sampling frequency to 1    
    g = impulse(md,Ng); % Get the impulse response
    g = g(2:end); % Remove the first direct term in the impulse response
    u = randn(N,1); % Construct a random input with unit variance
    y = lsim(md,u); % Simulate the output        
    SNR_s = rand()*9+1;
    e = sqrt((var(y)/SNR_s))*randn(N,1); % Construct random white measurement noise    
    y = y + e; % Add measurement noise
    
    % Store the data         
    U(i,:) = u;
    Y(i,:) = y;
    G(i,:) = g;
    SNR(i) = SNR_s;
    
    

    if plot_data % Plot the result if desired
        figure(1)
        
        subplot(2,1,1)
        plot(u)
        title('input')
        hold on
        
        subplot(2,1,2)
        plot(y)
        title('output')
        hold on    
        
        figure(2)               
        bodemag(md)
        hold on
        title('Frequency response')
        
        figure(3)        
        plot(g)
        hold on
        title('Impulse response')
        pause
    end
end

if save_data == 1 % Save the data if desired
    save('data_new.mat','U','Y','G','SNR');
end
        
