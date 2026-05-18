clc;
clear all;
close all;

%% Parameters
f_0                  = 3.6192e9;                % centre frequency
c                    = 3e8;                     % speed of light in m/s
PRB                  = 106;                     % number of Physical Resource Blocks
delta_f              = 30e3;                    % subcarrier spacing
BW                   = PRB*12*delta_f;          % bandwidth
N                    = 1536;                    % fft size
total_carriers       = PRB*12;                  % number of sucarriers of the system
guard_carriers       = N - total_carriers;      % number of Guard subcarriers
f_s                  = N*delta_f;               % sampling rate
%% srs configuration
carrier              = nrCarrierConfig('SubcarrierSpacing',delta_f/1e3,'NSizeGrid',PRB);
srs                  = nrSRSConfig;
srs.CSRS             = 25;
srs.SymbolStart      = 0;
info                 = nrOFDMInfo(carrier);
ind                  = nrSRSIndices(carrier,srs);
sym                  = nrSRS(carrier,srs);
slotGrid             = zeros(total_carriers,1);
slotGrid(ind)        = sym;
num_srs              = length(sym);
ind                  = double(ind);

% Subcarrier indices of the SRS and noise after skipping DC subcarriers
ind_skip_dc          = guard_carriers/2 + ((PRB/2)*12+1:((PRB/2)+1)*12);
srs_subcarrier_idx   = [guard_carriers/2 + ind(1):2:ind_skip_dc(1)-1 ind_skip_dc(end)+1:2:guard_carriers/2 + ind(end)];
noise_subcarrier_idx = [(guard_carriers/2 + ind(1)) + 1:2:ind_skip_dc(1)-1 ind_skip_dc(end)+2:2:guard_carriers/2 + ind(end)];

%% IQ extraction from the recorded files
folder       = "ssb_measurements/";
sub_folder   = "scenario_ssb_10m_ue_att_20_20240405T135125/"; % replace with the name of the folder

fileID    = fopen(folder+sub_folder+'srs_chT.raw'); % open the file
IQ        = fread(fileID,'int16');
I         = IQ(1:2:end);                            % separate I
Q         = IQ(2:2:end);                            % separate Q
IQ        = I+1j*Q;                                 % Form a complex signal
srs_chT   = reshape(IQ,N,[]);
fclose(fileID);                                     % close file

fileID    = fopen(folder+sub_folder+'srs_chF.raw'); % open the file
IQ        = fread(fileID,'int16');
I         = IQ(1:2:end);                            % separate I
Q         = IQ(2:2:end);                            % separate Q
IQ        = I+1j*Q;                                 % Form a complex signal
srs_chF   = reshape(IQ,N,[]);
fclose(fileID);

fileID    = fopen(folder+sub_folder+'srsrxdataF.raw'); % open the file
IQ        = fread(fileID,'int16');
I         = IQ(1:2:end);                            % separate I
Q         = IQ(2:2:end);                            % separate Q
IQ        = I+1j*Q;                                 % Form a complex signal
srsrxdataF= reshape(IQ,N,[]);
fclose(fileID);

fileID    = fopen(folder+sub_folder+'srs_chF_lin_interp.raw');    % open the file
IQ        = fread(fileID,'int16');
I         = IQ(1:2:end);                            % separate I
Q         = IQ(2:2:end);                            % separate Q
IQ        = I+1j*Q;                                 % Form a complex signal
srs_chF_lin_interp= reshape(IQ,N,[]);
fclose(fileID);

%% Removing initial invalid samples
srs_chT              = srs_chT(:,500:end);
srs_chF              = fftshift(srs_chF(:,500:end));
srsrxdataF           = fftshift(srsrxdataF(:,500:end));
srs_chF_lin_interp   = srs_chF_lin_interp(:,500:end);
%% SNR Estimation
% SNR = (|Y| - |N|)/|N|
signal_subcarriers   = srsrxdataF(srs_subcarrier_idx,:);
noise_subcarriers    = srsrxdataF(noise_subcarrier_idx,:);              
signal_power         = mean(abs(signal_subcarriers).^2);
noise_power          = mean(abs(noise_subcarriers).^2);
SNR_est_lin          = (mean(signal_power)-mean(noise_power))/mean(noise_power);
SNR_est_dB           = 10*log10(SNR_est_lin);
%% Plot Impulse response
figure();
stem(-N/2+1:N/2,abs(srs_chT(:,1))/2^15); % Divide by 2^15 to convert from fixed point to floating point
ylabel('Channel gain');
xlabel('Time index');

%% Plot Frequency response
figure();
stem(-N/2+1:N/2,abs(srs_chF(:,1))/2^15); % Divide by 2^15 to convert from fixed point to floating point
ylabel('Channel gain');
xlabel('Subcarrier index');