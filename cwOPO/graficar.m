clc; clear all; %close all;
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');  
set(0,'defaultAxesFontSize',20)

for lb = [0 1]

    folder = 'fixed_files';

    T =load([folder,'/T.dat']);
    dT = T(2)-T(1);
    Tp =load([folder,'/Tp.dat']);
    freq =(load([folder,'/freq.dat']));

    folder = ['MgOPPLN_N_4_beta_0.8_df_LB_', num2str(lb)];

    sr = load([folder,'/signal_total_delta_0.000000_r.dat']);
    si = load([folder,'/signal_total_delta_0.000000_i.dat']);


    s  = sr(end-length(T)+1:end) + 1j*si(end-length(T)+1:end);
    sw = ifftshift(ifft(sr+1j*si));

    h = figure('units','normalized','outerposition',[0 0 1 1]);
    subplot(2,1,1)
    % yyaxis left
    hold on
    plot(T, abs(s).^2/max(abs(s).^2))

    subplot(2,1,2)
    hold on
    % area(PG_GDD(:,1),PG_GDD(:,2)/max(PG_GDD(:,2)), 'FaceAlpha', 0.3)
    plot(freq, abs(sw).^2/max(abs(sw).^2))
    xlim([-20,20])
end