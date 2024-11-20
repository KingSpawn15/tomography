clear all;

%%
% setup electron
utem_parameters.electron_total_energy = 1.1;
utem_parameters.electron_total_energy = 0.94;
utem_parameters.electron_total_time_fs = 360;
utem_parameters.electron_time_coherent_fwhm_fs = 50;
utem_parameters.electron_theta = -7*pi/180;
utem_parameters.electron_velocity_c = 0.7;
elec = UTEMElectron(utem_parameters);

% sample fields
vec_t = -0.5:0.01:3;
vec_z = -200:1:200;
[T, Z] = ndgrid(vec_t, vec_z);
sigma_z = 40;
sigma_t = 0.01;
Etest = (exp(-T.^2/(2*sigma_t^2)) .* exp(-Z.^2/(2 * sigma_z^2))).';

%% loss from lorentzian approach
[delta_t, eels_lorentzian] = lorentzian_loss_spectrum_from_fields(Etest, T, Z, utem_parameters.electron_velocity_c);

%% comparing with theoretical
vel = 0.7 * 3*10^(8+6-12);
theoretical_eels = calculate_theoretical_eels(delta_t, sigma_t, sigma_z, vel);

figure;
plot(delta_t, theoretical_eels, delta_t, eels_lorentzian,'r--');
legend('theoretical','computed');
xlabel('\Delta t [ps]')
ylabel('loss [eV]')


%% add broadening

e_w = linspace(-6,6,181);
psi_coherent = spectrum_to_coherent_eels_mod(e_w, delta_t, eels_lorentzian);
w = elec.incoherent_gaussian_blurring_window(e_w,delta_t);
psi_incoherent =  incoherent_convolution(psi_coherent, w, delta_t, e_w);

%% Create a tiled layout with square plots and adjusted figure size
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.4]); % Adjust figure size

tiledlayout(1, 2, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% Plot psi_coherent
nexttile;
imagesc(e_w, delta_t, psi_coherent);
xlabel('Energy Loss [eV]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('\Delta t [ps]', 'FontSize', 12, 'FontWeight', 'bold');
title('\psi_{coherent}', 'FontSize', 14, 'FontWeight', 'bold');
% colorbar;
xticks([-6:2:6]);
axis xy; % Correct orientation
axis square; % Make the plot square
set(gca, 'FontSize', 12, 'FontName', 'Arial');

% Plot psi_incoherent
nexttile;
imagesc(e_w, delta_t, psi_incoherent);
xlabel('Energy Loss [eV]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('\Delta t [ps]', 'FontSize', 12, 'FontWeight', 'bold');
title('\psi_{incoherent}', 'FontSize', 14, 'FontWeight', 'bold');
xticks([-6:2:6]);
colorbar;
axis xy; % Correct orientation
axis square; % Make the plot square
set(gca, 'FontSize', 12,  'FontName', 'Arial');



function theoretical_eels = calculate_theoretical_eels(Delta_t, sigma_t, sigma_z, vel)


    % Calculate the numerator
    numerator = exp(- (vel^2 * Delta_t.^2) ./ (2 * (vel^2 * sigma_t^2 + sigma_z^2))) * sqrt(2 * pi);

    % Calculate the denominator
    denominator = sqrt(1 / (vel^2 * sigma_t^2) + 1 / sigma_z^2);

    % Compute the result
    theoretical_eels = numerator ./ denominator;
end

function [t0_vec, eels] = lorentzian_loss_spectrum_from_fields(EOR, TOR, ZOR, velec)

% Units 
% E [V/um]
% Z [um]
% T [ps]
% velec[units of c]

EOR = EOR.';
max_z = 50;
velec = velec * 3 * 10^(8+6-12);

electron_z = (-3 : 0.01: 3).' .* max_z; % pay attention here these are the limits of the integral
electron_t0 = electron_z / velec;
t0_vec = (-2.3:0.01:2.3).';
eels = zeros(length(t0_vec),1);


length_t0_vec = length(t0_vec);

parfor ind = 1: length_t0_vec
  
    et_l = interp2(TOR', ZOR', EOR', electron_t0 + t0_vec(ind), electron_z, "linear",0);
    eels(ind) = trapz(electron_z,et_l);

end


end

function psi_coherent = spectrum_to_coherent_eels_mod(e_w, t0_vec, loss_spectrum)
    
%     loss_spectrum = interp1(t0_vec.',loss_spectrum.',t_w,'linear',0);
    
    t_w = t0_vec;
    eels_w = loss_spectrum;
    eels_w = repmat(eels_w,[1,length(e_w)]);
    
%     e_w = linspace(-5,5,181);
    e_w_mat = e_w;
    e_w_mat = abs(repmat(e_w_mat,[length(t_w),1]) - eels_w);
    
    [~, b] = min(e_w_mat,[],2);
    
    psi_coherent = zeros(length(t_w), length(e_w));
    row = 1;
    for col = b.'
        psi_coherent(row,col) = 1;
        row = row + 1;
    end

    
end

function psi_incoherent = incoherent_convolution(psi_coherent, w, t_w, e_w ,...
        w_cut_off_factor)
    
    if nargin < 5
        w_cut_off_factor = 0.01;
    end
    
    psi_sum = zeros(size(psi_coherent));
    
    w_cutOff = w_cut_off_factor*max(w(:));
    
    % Example dont use such code. It becomes hard to modify
    % w_cutOff = 0.01*max(w(:));
    
    parfor t_ind = 1:length(t_w)
        for e_ind = 1:length(e_w)
            
            if w(t_ind,e_ind) < w_cutOff
                continue
            end
            
            psi_sum = psi_sum + w(t_ind,e_ind).*...
                circshift(circshift(psi_coherent,-ceil(length(t_w)/2)+t_ind,1),-ceil(length(e_w)/2)+e_ind,2);
            
        end
    end
    
    psi_coherent = psi_sum;
    psi_incoherent = psi_coherent./trapz(e_w,psi_coherent,2);
    psi_incoherent = psi_incoherent./max(psi_incoherent(:));
end
