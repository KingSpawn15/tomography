classdef UTEMElectron < handle
    %UTEMELECTRON Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        electron_total_energy
        electron_total_time
        electron_time_coherent_fwhm
        electron_theta
        electron_velocity
        electron_time_incoherent_sigma
        electron_energy_incoherent_sigma
        electron_time_coherent_sigma
    end
    
    methods
        
        function self = UTEMElectron(utem_parameters)
            %UTEMELECTRON Construct an instance of this class
            %   Detailed explanation goes here
            
            self.electron_total_energy = utem_parameters.electron_total_energy;
            self.electron_total_time = utem_parameters.electron_total_time_fs * 1e-15;
            self.electron_time_coherent_fwhm = utem_parameters.electron_time_coherent_fwhm_fs * 1e-15;
            self.electron_theta = utem_parameters.electron_theta;
            self.electron_velocity = utils.c2msec(utem_parameters.electron_velocity_c);
            self.electron_time_coherent_sigma = self.parameters_coherent();
            [self.electron_time_incoherent_sigma,...
                self.electron_energy_incoherent_sigma] = self.parameters_incoherent();

            
        end
        
        
        function electron_time_coherent_sigma = parameters_coherent(self)
            
            electron_time_coherent_sigma = self.electron_time_coherent_fwhm./(2*sqrt(2*log(2)));%[s]
            
        end
        
        
        function [electron_time_incoherent_sigma,...
                electron_energy_incoherent_sigma] = parameters_incoherent(self)
            
            %Incoherent parameters
            electron_time_incoherent_sigma = self.electron_total_time/(2*sqrt(2*log(2)));%[s]
            electron_energy_incoherent_sigma = self.electron_total_energy/(2*sqrt(2*log(2)));%[eV]
            
        end
        
        function [w, e_w, t_w] = energy_time_grid(self, sub_sample_factor, energy, deltat)
            
            e_w = energy(1:sub_sample_factor:end);
            t_w = deltat*1e12;%[ps]
            
            sigma_t = self.electron_time_incoherent_sigma*1e12;
            sigma_e= self.electron_energy_incoherent_sigma;
            
            a = cos(self.electron_theta)^2/(2*sigma_t^2) + sin(self.electron_theta)^2/(2*sigma_e^2);
            b = (sin(2*self.electron_theta)/4)*((1/sigma_e^2)-(1/sigma_t^2));
            c = sin(self.electron_theta)^2/(2*sigma_t^2) + cos(self.electron_theta)^2/(2*sigma_e^2);
            
            [TW,EW] = meshgrid(t_w,e_w);
            w = exp(-(a*TW.^2 + 2*b*TW.*EW + c*EW.^2));
            w = w';
        end
        
        function w = incoherent_gaussian_blurring_window(self, e_w, t_w)
            
%             e_w = energy(1:sub_sample_factor:end);
%             t_w = deltat*1e12;%[ps]
            
            sigma_t = self.electron_time_incoherent_sigma*1e12;
            sigma_e= self.electron_energy_incoherent_sigma;
            
            a = cos(self.electron_theta)^2/(2*sigma_t^2) + sin(self.electron_theta)^2/(2*sigma_e^2);
            b = (sin(2*self.electron_theta)/4)*((1/sigma_e^2)-(1/sigma_t^2));
            c = sin(self.electron_theta)^2/(2*sigma_t^2) + cos(self.electron_theta)^2/(2*sigma_e^2);
            
            [TW,EW] = meshgrid(t_w,e_w);
            w = exp(-(a*TW.^2 + 2*b*TW.*EW + c*EW.^2));
            w = w';
        end

    end
    
    methods(Static)
        
        
        
    end
    
end

