clc; clear all; %close all;
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');  
set(0,'defaultAxesFontSize',20)

cc = 1;
for dd = 0
    for gdd = [0 50 90 ]
        h = figure('units','normalized','outerposition',[0 0 1 1]);
        
        
        folder = ['cw_PPLN_R_0.98_cav_length_5_delta_',...
            num2str(dd),'_Power_2_W_GDD_',num2str(gdd),'_ModDepth_05'];
        
        T =load([folder,'/T.dat']);
        Tp=load([folder,'/Tp.dat']);
        Tp = (Tp-min(Tp))/max(T);
        
        freq=load([folder,'/freq.dat']);

        signal_r=load([folder,'/signal_total_delta_',num2str(dd/1000,'%.6f'),'_r.dat']);
        signal_i=load([folder,'/signal_total_delta_',num2str(dd/1000,'%.6f'),'_i.dat']);

        SIGNAL = signal_r + 1j*signal_i;

        subplot(2,1,1)
        plot(Tp,abs(SIGNAL).^2)
        title(['GDD = ',num2str(gdd), '%'])
        subplot(2,1,2)
        hold on
        plot(freq,abs(ifftshift(ifft(SIGNAL))).^2)
%         title(['$\delta$ = ',num2str(dd/1000),'$\pi$'])
%         ylim([0,0.03])
        box on;        grid on;
    end
    
    cc = cc+1;
end

% h = figure('units','normalized','outerposition',[0 0 1 1]);
% 
% subplot(1,2,1)
% Field2_r=load([folder,'/Field2Dp_input_r.dat']);
% Field2_i=load([folder,'/Field2Dp_input_i.dat']);
% Field2 = Field2_r + 1j*Field2_i;
% imagesc(abs(Field2).^2)
% title(['Input'])
% 
% subplot(1,2,2)
% Field2_r=load([folder,'/Field2Dp_output_r.dat']);
% Field2_i=load([folder,'/Field2Dp_output_i.dat']);
% Field2 = Field2_r + 1j*Field2_i;
% imagesc(abs(Field2).^2)
% title(['Output'])
% 

% h = figure('units','normalized','outerposition',[0 0 1 1]);
% 
% PropagPr=load([folder,'/prueba_r.dat']);
% PropagPi=load([folder,'/prueba_i.dat']);
% Field2 = PropagPr + 1j*PropagPi;
% subplot(1,3,1)
% imagesc(real(Field2))
% subplot(1,3,2)
% imagesc(imag(Field2))
% subplot(1,3,3)
% imagesc(abs(Field2).^2)
