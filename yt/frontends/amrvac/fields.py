"""
AMRVAC-specific fields

"""


import functools

import numpy as np
import sympy as sp

from yt.frontends.amrvac.physics import ionisation

from yt.fields.field_info_container import FieldInfoContainer
from yt.fields.magnetic_field import setup_magnetic_field_aliases
from yt.units import dimensions
from yt.utilities.logger import ytLogger as mylog

import yt.utilities.physical_constants as pc

# We need to specify which fields we might have in our dataset.  The field info
# container subclass here will define which fields it knows about.  There are
# optionally methods on it that get called which can be subclassed.

direction_aliases = {
    "cartesian": ("x", "y", "z"),
    "polar": ("r", "theta", "z"),
    "cylindrical": ("r", "z", "theta"),
    "spherical": ("r", "theta", "phi"),
}

def _velocity(field, data, idir, prefix=None):
    """Velocity = linear momentum / density"""
    # This is meant to be used with functools.partial to produce
    # functions with only 2 arguments (field, data)
    # idir : int
    #    the direction index (1, 2 or 3)
    # prefix : str
    #    used to generalize to dust fields
    if prefix is None:
        prefix = ""
    moment = data["gas", "%smoment_%d" % (prefix, idir)]
    rho = data["gas", f"{prefix}density"]

    mask1 = rho == 0
    if mask1.any():
        mylog.info(
            "zeros found in %sdensity, "
            "patching them to compute corresponding velocity field.",
            prefix,
        )
        mask2 = moment == 0
        if not ((mask1 & mask2) == mask1).all():
            raise RuntimeError
        rho[mask1] = 1
    return moment / rho


code_density = "code_mass / code_length**3"
code_moment = "code_mass / code_length**2 / code_time"
code_pressure = "code_mass / code_length / code_time**2"


class AMRVACFieldInfo(FieldInfoContainer):
    # for now, define a finite family of dust fields (up to 100 species)
    MAXN_DUST_SPECIES = 100
    known_dust_fields = [
        ("rhod%d" % idust, (code_density, ["dust%d_density" % idust], None))
        for idust in range(1, MAXN_DUST_SPECIES + 1)
    ] + [
        (
            "m%dd%d" % (idir, idust),
            (code_moment, ["dust%d_moment_%d" % (idust, idir)], None),
        )
        for idust in range(1, MAXN_DUST_SPECIES + 1)
        for idir in (1, 2, 3)
    ]
    # format: (native(?) field, (units, [aliases], display_name))
    # note: aliases will correspond to "gas" typed fields
    # whereas the native ones are "amrvac" typed
    known_other_fields = (
        ("rho", (code_density, ["density"], None)),
        ("m1", (code_moment, ["moment_1"], None)),
        ("m2", (code_moment, ["moment_2"], None)),
        ("m3", (code_moment, ["moment_3"], None)),
        ("e", (code_pressure, ["energy_density"], None)),
        ("eaux", (code_pressure, ["internal_energy_density"], None)),
        ("b1", ("code_magnetic", ["magnetic_1"], None)),
        ("b2", ("code_magnetic", ["magnetic_2"], None)),
        ("b3", ("code_magnetic", ["magnetic_3"], None)),
        ("Te", ("code_temperature", ["temperature"], None)),
        *known_dust_fields,
    )

    known_particle_fields = ()

    def _setup_velocity_fields(self, idust=None):
        if idust is None:
            dust_flag = dust_label = ""
        else:
            dust_flag = "d%d" % idust
            dust_label = "dust%d_" % idust

        us = self.ds.unit_system
        for idir, alias in enumerate(direction_aliases[self.ds.geometry], start=1):
            if not ("amrvac", "m%d%s" % (idir, dust_flag)) in self.field_list:
                break
            velocity_fn = functools.partial(_velocity, idir=idir, prefix=dust_label)
            functools.update_wrapper(velocity_fn, _velocity)
            self.add_field(
                ("gas", f"{dust_label}velocity_{alias}"),
                function=velocity_fn,
                units=us["velocity"],
                dimensions=dimensions.velocity,
                sampling_type="cell",
            )
            self.alias(
                ("gas", "%svelocity_%d" % (dust_label, idir)),
                ("gas", f"{dust_label}velocity_{alias}"),
                units=us["velocity"],
            )
            self.alias(
                ("gas", f"{dust_label}moment_{alias}"),
                ("gas", "%smoment_%d" % (dust_label, idir)),
                units=us["density"] * us["velocity"],
            )

    def _setup_dust_fields(self):
        idust = 1
        imax = self.__class__.MAXN_DUST_SPECIES
        while ("amrvac", "rhod%d" % idust) in self.field_list:
            if idust > imax:
                mylog.error(
                    "Only the first %d dust species are currently read by yt. "
                    "If you read this, please consider issuing a ticket. ",
                    imax,
                )
                break
            self._setup_velocity_fields(idust)
            idust += 1
        n_dust_found = idust - 1

        us = self.ds.unit_system
        if n_dust_found > 0:

            def _total_dust_density(field, data):
                tot = np.zeros_like(data[("gas", "density")])
                for idust in range(1, n_dust_found + 1):
                    tot += data["dust%d_density" % idust]
                return tot

            self.add_field(
                ("gas", "total_dust_density"),
                function=_total_dust_density,
                dimensions=dimensions.density,
                units=us["density"],
                sampling_type="cell",
            )

            def dust_to_gas_ratio(field, data):
                return data[("gas", "total_dust_density")] / data[("gas", "density")]

            self.add_field(
                ("gas", "dust_to_gas_ratio"),
                function=dust_to_gas_ratio,
                dimensions=dimensions.dimensionless,
                sampling_type="cell",
            )

    def _setup_b0split_fields(self):
        def _b0i_field_split(field, data, idir):
            # This is meant to be used with functools.partial and magnetic field splitting.
            # The sympy-expressions are lambified and evaluated over the grid, and the total
            # magnetic field is returned (instead of the perturbed one).
            # idir : int
            #   the direction index (1, 2 or 3)

            b0i = sp.lambdify(
                data.ds._allowed_b0split_symbols, data.ds._b0field.get(f"b0{idir}"),
                "numpy"
            )(
                data["index", "x"].value, 
                data["index", "y"].value, 
                data["index", "z"].value
            )
            b0i_total = b0i * data.ds.magnetic_unit + data["amrvac", f"b{idir}"].to(
                data.ds.magnetic_unit
            )
            return b0i_total

        mylog.info("Setting up fields with magnetic field splitting enabled.")
        for idir in "123":
            if ("amrvac", f"b{idir}") not in self.ds.field_list:
                break
            b0i_field_split_fn = functools.partial(_b0i_field_split, idir=idir)
            functools.update_wrapper(b0i_field_split_fn, _b0i_field_split)
            # this will override the default fields ("gas", "magnetic_i")
            # dimensionful analog
            self.add_field(
                ("gas", f"magnetic_{idir}"),
                function=b0i_field_split_fn,
                sampling_type="cell",
                units=self.ds.unit_system["magnetic_field_cgs"],
                dimensions=dimensions.magnetic_field_cgs,
                force_override=True,
            )
            # make alias for magnetic code unit analogs
            self.alias(
                ("amrvac", f"magnetic_{idir}"),
                ("gas", f"magnetic_{idir}"),
                units="code_magnetic",
            )

        # override energy density, for B0-splitted datasets only the perturbed
        # energy e1 is saved to the datfile. Total energy density is then given by
        # etot = e_internal + e_kinetic + e_magnetic
        if ('amrvac','eaux') in self.ds.field_list:
            def _etot_field_split(field, data):
                return (data["gas", "internal_energy_density"]
                    + data["gas","kinetic_energy_density"]
                    + data["gas","magnetic_energy_density"]
                    )

            # force override default e field
            self.add_field(
                ("gas", "energy_density"),
                function=_etot_field_split,
                sampling_type="cell",
                units=self.ds.unit_system["density"] * self.ds.unit_system["velocity"] ** 2,
                dimensions=dimensions.density * dimensions.velocity ** 2,
                force_override=True,
            )
            # alias for code unit analog
            self.alias(
                ("amrvac", "energy_density"),
                ("gas", "energy_density"),
                units="code_pressure",
            )
        else:
            mylog.warning("eaux not provided in bsplit .dat, unable to recover true energy")
            mylog.info("Defaulting back to perturbed energy")

    def setup_fluid_fields(self):
        setup_magnetic_field_aliases(self, "amrvac", [f"mag{ax}" for ax in "xyz"])
        self._setup_velocity_fields()  # gas velocities
        self._setup_dust_fields()  # dust derived fields (including velocities)
        # for magnetic field splitting, overrides default
        # ("gas", "magnetic_i") and e fields
        if self.ds._b0_is_split:
            self._setup_b0split_fields()

        # fields with nested dependencies are defined thereafter
        # by increasing level of complexity
        us = self.ds.unit_system

        def _kinetic_energy_density(field, data):
            # devnote : have a look at issue 1301
            return 0.5 * data["gas", "density"] * data["gas", "velocity_magnitude"] ** 2

        self.add_field(
            ("gas", "kinetic_energy_density"),
            function=_kinetic_energy_density,
            units=us["density"] * us["velocity"] ** 2,
            dimensions=dimensions.density * dimensions.velocity ** 2,
            sampling_type="cell",
        )

        # magnetic energy density
        if ("amrvac", "b1") in self.field_list:

            def _magnetic_energy_density(field, data):
                emag = 0.5 * data["gas", "magnetic_1"] ** 2
                for idim in "23":
                    if not ("amrvac", f"b{idim}") in self.field_list:
                        break
                    emag += 0.5 * data["gas", f"magnetic_{idim}"] ** 2
                # in AMRVAC the magnetic field is defined in units where mu0 = 1,
                # such that
                # Emag = 0.5*B**2 instead of Emag = 0.5*B**2 / mu0
                # To correctly transform the dimensionality from gauss**2 -> rho*v**2,
                # we have to take mu0 into account. If we divide here, units when adding
                # the field should be us["density"]*us["velocity"]**2.
                # If not, they should be us["magnetic_field"]**2 and division should
                # happen elsewhere.
                emag /= 4 * np.pi
                # divided by mu0 = 4pi in cgs,
                # yt handles 'mks' and 'code' unit systems internally.
                return emag

            self.add_field(
                ("gas", "magnetic_energy_density"),
                function=_magnetic_energy_density,
                units=us["density"] * us["velocity"] ** 2,
                dimensions=dimensions.density * dimensions.velocity ** 2,
                sampling_type="cell",
            )

        # Adding the thermal pressure field.
        # In AMRVAC we have multiple physics possibilities:
        # - if HD/MHD + energy equation P = (gamma-1)*(e - ekin (- emag)) for (M)HD
        # - if HD/MHD but solve_internal_e is true in parfile, P = (gamma-1)*e for both
        # - if (m)hd_energy is false in parfile (isothermal), P = c_adiab * rho**gamma

        def _full_thermal_pressure_HD(field, data):
            # energy density and pressure are actually expressed in the same unit
            pthermal = (data.ds.gamma - 1) * (
                data["gas", "energy_density"] - data["gas", "kinetic_energy_density"]
            )
            return pthermal

        def _full_thermal_pressure_MHD(field, data):
            pthermal = (
                _full_thermal_pressure_HD(field, data)
                - (data.ds.gamma - 1) * data["gas", "magnetic_energy_density"]
            )
            return pthermal

        def _polytropic_thermal_pressure(field, data):
            return (data.ds.gamma - 1) * data["gas", "energy_density"]

        def _adiabatic_thermal_pressure(field, data):
            return data.ds._c_adiab * data["gas", "density"] ** data.ds.gamma

        pressure_recipe = None
        if ("amrvac", "e") in self.field_list:
            if self.ds._e_is_internal:
                pressure_recipe = _polytropic_thermal_pressure
                mylog.info("Using polytropic EoS for thermal pressure.")
            elif ("amrvac", "b1") in self.field_list:
                pressure_recipe = _full_thermal_pressure_MHD
                mylog.info("Using full MHD energy for thermal pressure.")
            else:
                pressure_recipe = _full_thermal_pressure_HD
                mylog.info("Using full HD energy for thermal pressure.")
        elif self.ds._c_adiab is not None:
            pressure_recipe = _adiabatic_thermal_pressure
            mylog.info("Using adiabatic EoS for thermal pressure (isothermal).")
            mylog.warning(
                "If you used usr_set_pthermal you should "
                "redefine the thermal_pressure field."
            )

        if pressure_recipe is not None:
            self.add_field(
                ("gas", "thermal_pressure"),
                function=pressure_recipe,
                units=us["density"] * us["velocity"] ** 2,
                dimensions=dimensions.density * dimensions.velocity ** 2,
                sampling_type="cell",
            )
            self.alias(
                ("amrvac", "thermal_pressure"),
                ("gas", "thermal_pressure"),
                units="code_pressure",
            )

            # sound speed and temperature depend on thermal pressure
            def _sound_speed(field, data):
                return np.sqrt(
                    data.ds.gamma
                    * data["gas", "thermal_pressure"]
                    / data["gas", "density"]
                )

            self.add_field(
                ("gas", "sound_speed"),
                function=_sound_speed,
                units=us["velocity"],
                dimensions=dimensions.velocity,
                sampling_type="cell",
            )
        else:
            mylog.warning(
                "e not found and no parfile passed, can not set thermal_pressure."
            )

        
        if self.ds.ionisation:

            def _nlte_ion(field, data):
                # mylog.info('ionisation altitude: '+str(data.ds.altitude)+' km')
                # ionisation.init_splines(self.ds.altitude)
                return ionisation.spline_ion.ev(data["gas","temperature"], data["gas","thermal_pressure"])*data.ds.units.dimensionless

            self.add_field(
                ("gas", "nlte_ion"),
                function=_nlte_ion,
                units=us["dimensionless"],
                dimensions=dimensions.dimensionless,
                sampling_type="cell",
            )

            def _fpar(field, data):
                # mylog.info('ionisation altitude: '+str(data.ds.altitude)+' km')
                # ionisation.init_splines(self.ds.altitude)
                return ionisation.spline_f.ev(data["gas","temperature"], data["gas","thermal_pressure"])*(data.ds.units.cm**-3)

            self.add_field(
                ("gas", "fpar"),
                function=_fpar,
                units=us["length"]**-3,
                dimensions=dimensions.length**-3,
                sampling_type="cell",
            )

            def _get_ne(field, data):
                return data["gas","thermal_pressure"] / ((1 + 1.1 / data['gas','nlte_ion'])
                    * pc.boltzmann_constant_cgs 
                    * data["gas","temperature"])

            self.add_field(
                ("gas", "electron_number_density"),
                function=_get_ne,
                units=us["length"]**-3,
                dimensions=dimensions.length**-3,
                sampling_type="cell",
            )

            def _get_saha_ne(field, data):
                mass_hydrogen=1.67e-24
                mass_electron=9.11e-28
                k_b=pc.boltzmann_constant_cgs.value
                p_c=6.63e-27

                xsi_hydrogen=13.53*1.602e-12
                xsi_helium=24.48*1.602e-12
                xsi_heliumI=54.17*1.602e-12
                a_0=5.29177e-9 #Bohr radius in cm
                Ry=13.60569*1.602e-12
                N_T=1e11 #Total number density (assumed)

                Z_eff=1.0
                q=np.sqrt(Z_eff/(2.0*np.pi*a_0),dtype=float)*np.power(N_T,-1.0/6.0,dtype=float)
                n_max=0.5*q*(1.0+np.sqrt(1.0+4.0/q,dtype=float))
                E_cut=(13.59844*1.602e-12)-(np.square(Z_eff,dtype=float)*Ry/np.power(n_max,2.0,dtype=float))
                U_H=2.0*np.exp(-Ry/(k_b*data['temperature'].value),dtype=float)+278.0*np.exp(-150991.49/data['temperature'].value,dtype=float)+((2.0*(np.power(n_max,3.0,dtype=float) -343.0)/3.0)*np.exp(-E_cut/(k_b*data['temperature'].value)))

                U_HI=1.0

                Z_eff=2.0
                q=np.sqrt(Z_eff/(2.0*np.pi*a_0),dtype=float)*np.power(N_T,-1.0/6.0,dtype=float)
                n_max=0.5*q*(1.0+np.sqrt(1.0+4.0/q,dtype=float))
                E_cut=(24.58741*1.602e-12)-(np.square(Z_eff,dtype=float)*Ry/np.power(n_max,2.0,dtype=float))
                U_He=1.0*np.exp(-1.78678*Ry/(k_b*data['temperature'].value),dtype=float)+556.0*np.exp(-278302.52/data['temperature'].value,dtype=float)+((4.0*(np.power(n_max,3.0,dtype=float) -343.0)/3.0)*np.exp(-E_cut/(k_b*data['temperature'].value)))

                Z_eff=2.0
                q=np.sqrt(Z_eff/(2.0*np.pi*a_0),dtype=float)*np.power(N_T,-1.0/6.0,dtype=float)
                n_max=0.5*q*(1.0+np.sqrt(1.0+4.0/q,dtype=float))
                E_cut=(54.41778*1.602e-12)-(np.square(Z_eff,dtype=float)*Ry/np.power(n_max,2.0,dtype=float))
                U_HeI=2.0*np.exp(-4.0*Ry/(k_b*data['temperature'].value),dtype=float)+278.0*np.exp(-604233.37/data['temperature'].value,dtype=float)+((2.0*(np.power(n_max,3.0,dtype=float) -343.0)/3.0)*np.exp(-E_cut/(k_b*data['temperature'].value)))

                U_HeII=1.0

                #Saha equations
                K_1 = 2.0 * (U_HI/U_H) * (((2.0 * np.pi * mass_electron * k_b * data['temperature'].value)**1.5)/(p_c**3))*np.exp(-xsi_hydrogen/(k_b*data['temperature'].value))

                K_2 = 2.0 * (U_HeI/U_He) * (((2.0 * np.pi * mass_electron * k_b * data['temperature'].value)**1.5)/(p_c**3))*np.exp(-xsi_helium/(k_b*data['temperature'].value))

                K_3 = 2.0 * (U_HeII/U_HeI) * (((2.0 * np.pi * mass_electron * k_b * data['temperature'].value)**1.5)/(p_c**3))*np.exp(-xsi_heliumI/(k_b*data['temperature'].value))

                #Iterative Solver

                nHe=data['gas','number_density'].value/10.
                nH=data['gas','number_density'].value
                
                deltae=1.e7
                n_e=1e11 # initial value
                while deltae>1e-4:
                    n_H=nH/(1.0+K_1/n_e) # neutral hydrogen number density
                    n_H1=nH-n_H # once-ionised hydrogen number density
                    n_He=nHe/(K_2*K_3/(n_e*n_e)+1.0+K_2/n_e) # neutral helium number density
                    n_He1=K_2/n_e*n_He # once-ionised helium number density
                    n_He2=nHe-n_He1-n_He # twice-ionised hydrogen number density
                    n_i=n_He1+n_He2+n_H1 # Ion density
                    deltae=abs(np.amax(n_He1+n_He2*2.0+n_H1-n_e) ) # check the charge neutrality condition
                    deltae=deltae/np.amin(n_He1+n_He2*2.0+n_H1) # check the charge neutrality condition
                    n_e=n_He1+n_He2*2.0+n_H1 # Electron density

                return n_e*data.ds.units.cm**-3

            self.add_field(
                ("gas", "saha_electron_number_density"),
                function=_get_saha_ne,
                units=us["length"]**-3,
                dimensions=dimensions.length**-3,
                sampling_type="cell",
            )

            def _halpha_abcof(field, data): #Absorption coefficient
                f23 = 0.6407
                n2 = data['gas','electron_number_density'].to('cm**-3')**2 / (data['gas','fpar'] * 1e16) #fpar table in multiples of 1e16
                ksi = 5 * 1e5 *data.ds.units.cm/data.ds.units.s  # microturbulence in cm/s
                nu_0 = (pc.c / (6562.8 * 1e-8*data.ds.units.cm))       # H-alpha wavelength is 6562.8 Angstrom
                delta_nu = 0/data.ds.units.s
                delta_nuD = ((nu_0 / pc.c) * \
                            np.sqrt((2 * pc.boltzmann_constant_cgs * data['gas','temperature']) / pc.mp + ksi ** 2)).to('1/s')
                phi_nu = 1.0 / (np.sqrt(np.pi) * delta_nuD) * np.exp(-delta_nu / delta_nuD) ** 2
                abcof = (np.pi * pc.elementary_charge**2 / (pc.me * pc.c)) * \
                                        f23 * n2 * phi_nu
                return abcof

            self.add_field(
                ("gas", "halpha_absorption_coefficient"),
                function=_halpha_abcof,
                units=us["length"]**-1,
                dimensions=dimensions.length**-1,
                sampling_type="cell",
            )

            def _euv_abcof(field, data):

                A_H=10.0/11.0
                A_He=1.0/10.0
                
                mass_hydrogen=1.67e-24
                mass_electron=9.11e-28
                k_b=pc.boltzmann_constant_cgs.value
                p_c=6.63e-27

                xsi_hydrogen=13.53*1.602e-12
                xsi_helium=24.48*1.602e-12
                xsi_heliumI=54.17*1.602e-12
                a_0=5.29177e-9 #Bohr radius in cm
                Ry=13.60569*1.602e-12
                N_T=1e11 #Total number density (assumed)

                Z_eff=1.0
                q=np.sqrt(Z_eff/(2.0*np.pi*a_0),dtype=float)*np.power(N_T,-1.0/6.0,dtype=float)
                n_max=0.5*q*(1.0+np.sqrt(1.0+4.0/q,dtype=float))
                E_cut=(13.59844*1.602e-12)-(np.square(Z_eff,dtype=float)*Ry/np.power(n_max,2.0,dtype=float))
                U_H=2.0*np.exp(-Ry/(k_b*data['temperature'].value),dtype=float)+278.0*np.exp(-150991.49/data['temperature'].value,dtype=float)+((2.0*(np.power(n_max,3.0,dtype=float) -343.0)/3.0)*np.exp(-E_cut/(k_b*data['temperature'].value)))

                U_HI=1.0

                Z_eff=2.0
                q=np.sqrt(Z_eff/(2.0*np.pi*a_0),dtype=float)*np.power(N_T,-1.0/6.0,dtype=float)
                n_max=0.5*q*(1.0+np.sqrt(1.0+4.0/q,dtype=float))
                E_cut=(24.58741*1.602e-12)-(np.square(Z_eff,dtype=float)*Ry/np.power(n_max,2.0,dtype=float))
                U_He=1.0*np.exp(-1.78678*Ry/(k_b*data['temperature'].value),dtype=float)+556.0*np.exp(-278302.52/data['temperature'].value,dtype=float)+((4.0*(np.power(n_max,3.0,dtype=float) -343.0)/3.0)*np.exp(-E_cut/(k_b*data['temperature'].value)))

                Z_eff=2.0
                q=np.sqrt(Z_eff/(2.0*np.pi*a_0),dtype=float)*np.power(N_T,-1.0/6.0,dtype=float)
                n_max=0.5*q*(1.0+np.sqrt(1.0+4.0/q,dtype=float))
                E_cut=(54.41778*1.602e-12)-(np.square(Z_eff,dtype=float)*Ry/np.power(n_max,2.0,dtype=float))
                U_HeI=2.0*np.exp(-4.0*Ry/(k_b*data['temperature'].value),dtype=float)+278.0*np.exp(-604233.37/data['temperature'].value,dtype=float)+((2.0*(np.power(n_max,3.0,dtype=float) -343.0)/3.0)*np.exp(-E_cut/(k_b*data['temperature'].value)))

                U_HeII=1.0

                #Saha equations
                K_1 = 2.0 * (U_HI/U_H) * (((2.0 * np.pi * mass_electron * k_b * data['temperature'].value)**1.5)/(p_c**3))*np.exp(-xsi_hydrogen/(k_b*data['temperature'].value))

                K_2 = 2.0 * (U_HeI/U_He) * (((2.0 * np.pi * mass_electron * k_b * data['temperature'].value)**1.5)/(p_c**3))*np.exp(-xsi_helium/(k_b*data['temperature'].value))

                K_3 = 2.0 * (U_HeII/U_HeI) * (((2.0 * np.pi * mass_electron * k_b * data['temperature'].value)**1.5)/(p_c**3))*np.exp(-xsi_heliumI/(k_b*data['temperature'].value))

                #Iterative Solver

                nHe=data['gas','number_density'].value/10.
                nH=data['gas','number_density'].value
                
                deltae=1.e7
                n_e=1e11 # initial value
                while deltae>1e-4:
                    n_H=nH/(1.0+K_1/n_e) # neutral hydrogen number density
                    n_H1=nH-n_H # once-ionised hydrogen number density
                    n_He=nHe/(K_2*K_3/(n_e*n_e)+1.0+K_2/n_e) # neutral helium number density
                    n_He1=K_2/n_e*n_He # once-ionised helium number density
                    n_He2=nHe-n_He1-n_He # twice-ionised hydrogen number density
                    n_i=n_He1+n_He2+n_H1 # Ion density
                    deltae=abs(np.amax(n_He1+n_He2*2.0+n_H1-n_e) ) # check the charge neutrality condition
                    deltae=deltae/np.amin(n_He1+n_He2*2.0+n_H1) # check the charge neutrality condition
                    n_e=n_He1+n_He2*2.0+n_H1 # Electron density
                    

                X_H1=n_H1/nH
                X_He1=n_He1/nHe
                X_He2=n_He2/nHe


                nHe=data['gas','number_density']/10.0
                nH=data['gas','number_density']

                E=pc.planck_constant_cgs.to('eV*s').value*3e10/(int(data.ds.wavelength)*1e-8)

                #Taken from Keady & Kilcrease (2000) table 5.11
                a_TH=6.30
                a_THe=7.83
                a_THeI=1.58
                R_H=1.34
                R_He=1.66
                R_HeI=1.34
                E_TH=13.6
                E_THe=24.6
                E_THeI=54.4
                #Definition requires that E_H>E_T as this liberates the electron. 
                #Presumably liberation with zero residual momentum also occurs when values are equal.
                E_H=E
                E_He=E
                E_HeI=E
                S_H=2.99
                S_He=2.05
                S_HeI=2.99


                cross_s_H=a_TH*1e-18*((R_H*(E_TH/E_H)**S_H)+((1-R_H)*(E_TH/E_H)**(S_H+1)))*(data.ds.units.cm**2)
                cross_s_He=a_THe*1e-18*((R_He*(E_THe/E_He)**S_He)+((1-R_He)*(E_THe/E_He)**(S_He+1)))*(data.ds.units.cm**2)
                cross_s_HeI=a_THeI*1e-18*((R_HeI*(E_THeI/E_HeI)**S_HeI)+((1-R_HeI)*(E_THeI/E_HeI)**(S_HeI+1)))*(data.ds.units.cm**2)
                
                abcof=cross_s_H*A_H*(1-X_H1)
                abcof+=cross_s_He*(1.0-A_H)*(1.0-X_He1-X_He2)
                abcof+=cross_s_HeI*A_He*X_He1
                abcof=abcof*(nH+nHe)

                # abcof=cross_s_H*(data['gas','density'].to('g*cm**-3')/(1.1*pc.mp))
                # abcof+=cross_s_He*(0.1*data['gas','density'].to('g*cm**-3')/(1.1*pc.mp))
                # abcof+=cross_s_HeI*(0.1*data['gas','density'].to('g*cm**-3')/(1.1*pc.mp))

                return abcof

            self.add_field(
                ("gas", "euv_absorption_coefficient"),
                function=_euv_abcof,
                units=us["length"]**-1,
                dimensions=dimensions.length**-1,
                sampling_type="cell",
            )        

            def _emiss_jlambda(field,data):
                '''get optically thin emission coefficient in specific wave length'''
                

                def _findintable(Te,ne,logT,logn,gmat):
                    '''Find emissivity in perticular wave length or filter, given density and
                       temperature'''

                    def _findL(parr,aseq):
                        '''Find the location of a value in an ascending equi-distant sequence, 
                        return list of two indices'''
                        lenseq=len(aseq)
                        dx=aseq[1]-aseq[0]
                        jl=np.int32((parr-aseq[0])/dx)
                        jh=jl+1
                        jh[jl<0]=0
                        jl[jl<0]=0
                        jl[jh>lenseq-1]=lenseq-1
                        jh[jh>lenseq-1]=lenseq-1
                        return [jl,jh]
                    
                    Tel=np.log10(Te)
                    nel=np.log10(ne)
                    LT=_findL(Tel,logT)
                    Ln=_findL(nel,logn)

                    #bilinear interpolation in gmat(logT,logn) 2D table
                    dlog=logT[LT[1]]-logT[LT[0]]
                    dlog[dlog==0.]=1.
                    frac1=(logT[LT[1]]-Tel)/dlog
                    frac2=(Tel-logT[LT[0]])/dlog
                    frac1[LT[1]==LT[0]]=0.5
                    frac2[LT[1]==LT[0]]=0.5
                    #linear interpolation in logT
                    gr1=frac1*gmat[Ln[0],LT[0]]+frac2*gmat[Ln[0],LT[1]] 
                    gr2=frac1*gmat[Ln[1],LT[0]]+frac2*gmat[Ln[1],LT[1]]
                    dlog=logn[Ln[1]]-logn[Ln[0]]
                    dlog[dlog==0.]=1.
                    frac1=(logn[Ln[1]]-nel)/dlog
                    frac2=(nel-logn[Ln[0]])/dlog
                    frac1[Ln[1]==Ln[0]]=0.5
                    frac2[Ln[1]==Ln[0]]=0.5
                    corona_emiss=frac1*gr1+frac2*gr2 #linear interpolation in logn
                    return corona_emiss
                    

                # mylog.info("Synthesising "+self.ds.wavelength+" emission coefficient")
                

                tabledic = {'094': self.ds.euv_table_aia_094,
                            '131': self.ds.euv_table_aia_131,
                            '171': self.ds.euv_table_aia_171,
                            '193': self.ds.euv_table_aia_193,
                            '211': self.ds.euv_table_aia_211,
                            '304': self.ds.euv_table_aia_304,
                            '335': self.ds.euv_table_aia_335
                }

                if self.ds.wavelength=='304':
                    Ab=10.78/12.0
                else:
                    Ab=7.85/12.0
                
                #Table read removed to before the synthesis call    
                logT=tabledic[self.ds.wavelength]['logt']
                logn=tabledic[self.ds.wavelength]['n_e_lg']
                gmat=tabledic[self.ds.wavelength]['goft_mat']
                
                ce=_findintable(data['gas','temperature'].value,data['gas','saha_electron_number_density'].value,logT,logn,gmat)
                
                emiss = abs((Ab/(4.0*np.pi))*data['gas','saha_electron_number_density'].value**2 * ce)
                
                return emiss*data.ds.units.erg*data.ds.units.s**-1*data.ds.units.cm**-3*data.ds.units.sr**-1

            self.add_field(
                ("gas", "emission_coefficient"),
                function=_emiss_jlambda,
                units=self.ds.units.erg*self.ds.units.s**-1*self.ds.units.cm**-3*self.ds.units.sr**-1,
                dimensions=self.ds.units.erg*self.ds.units.s**-1*self.ds.units.cm**-3*self.ds.units.sr**-1,
                sampling_type="cell",
            )

            def _emiss_without_background(field,data):
                '''get optically thin emission in specific wave length'''        
                if self.ds.wavelength=='304':
                    emiss=data['gas','emission_coefficient']
                else: #adhoc integration assuming regions that contain high absorption will be at max level amr
                    emiss =(data['gas','emission_coefficient']*
                            (1.0-data['gas','euv_absorption_coefficient'].value*((self.ds.domain_width*self.ds.length_unit)/
                                                          (self.ds.domain_dimensions*self.ds.refine_by**self.ds.max_level)).value[0]))
                
                return emiss

            self.add_field(
                ("gas", "euv_emission"),
                function=_emiss_without_background,
                units=self.ds.units.erg*self.ds.units.s**-1*self.ds.units.cm**-3*self.ds.units.sr**-1,
                dimensions=self.ds.units.erg*self.ds.units.s**-1*self.ds.units.cm**-3*self.ds.units.sr**-1,
                sampling_type="cell",
            )    
