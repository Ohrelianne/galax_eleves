#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"
#include <xsimd/xsimd.hpp>
#include <omp.h>
#include <immintrin.h>
#include <iostream>

#define SERIAL 0
#define PARFOR_NAIVE 1
#define SERIAL_IMPROVED 2
#define XSIMD 3
#define XSIMD_OMP 4
#define XSIMD_OMP_OPTI 5

//SELECT THE STRATEGY YOU WANT TO USE
#define STRATEGY XSIMD_OMP_OPTI

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;
struct Rot
{
	static constexpr unsigned get(unsigned i, unsigned n)
	{
		return (i + n - 1) % n;
	}
};
Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}
std::vector<float> mask1={0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
std::vector<float> mask2={1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0};
std::vector<float> mask3={1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0};
std::vector<float> mask4={1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0};
std::vector<float> mask5={1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0};
std::vector<float> mask6={1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0};
std::vector<float> mask7={1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0};
std::vector<float> mask8={1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0};


std::vector<std::vector<float>> masks= {mask1,mask2,mask3,mask4,mask5,mask6,mask7,mask8};

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

#if STRATEGY == SERIAL 
for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;		for (std::size_t j = vec_size; j<n_particles; j++)
		{
			b_type diffx = particles.x[j] - rposx_i;
			b_type diffy = particles.y[j] - rposy_i;
			b_type diffz = particles.z[j] - rposz_i;	

			b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;	
			
			dij = xs::rsqrt(xs::max(once_v,dij)); //Il faut un batch de 1
			dij = 10.0 * dij * dij * dij;
			if (!((j < i) | (j>i+7))) {
				b_type mask = b_type::load_unaligned(&masks[j-i][0]);
				dij = dij * mask;
			}

			raccx_i = raccx_i + diffx * dij * initstate.masses[j];
			raccy_i = raccy_i + diffy * dij * initstate.masses[j];
			raccz_i = raccz_i + diffz * dij * initstate.masses[j];

			raccx_i.store_unaligned(&accelerationsx[i]);
			raccy_i.store_unaligned(&accelerationsy[i]);
			raccz_i.store_unaligned(&accelerationsz[i]);
    	}	
	}

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
				}

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}

	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}

#elif STRATEGY == SERIAL_IMPROVED 

for (int i = 0; i < n_particles; i ++)
    {
        for (int j = 0; j < i; j++)
        {
			const float diffx = particles.x[j] - particles.x[i];
			const float diffy = particles.y[j] - particles.y[i];
			const float diffz = particles.z[j] - particles.z[i];

			float dij = diffx * diffx + diffy * diffy + diffz * diffz;

			if (dij < 1.0)
			{
				dij = 10.0;
			}
			else
			{
				__m256 x_vec = _mm256_set1_ps(dij); // Passage en vecteur AVX
				__m256 inv_sqrt_x_vec = _mm256_rsqrt_ps(x_vec); // Calcul de l'inverse de la racine carrée
				inv_sqrt_x_vec = _mm256_mul_ps(inv_sqrt_x_vec,_mm256_mul_ps(inv_sqrt_x_vec,inv_sqrt_x_vec));// Multiplication
				float inv_sqrt_x = _mm256_cvtss_f32(inv_sqrt_x_vec);//reconversion en float
				dij = 10.0 * inv_sqrt_x;
			}
			
			accelerationsx[i] += diffx * dij * initstate.masses[j];
			accelerationsy[i] += diffy * dij * initstate.masses[j];
			accelerationsz[i] += diffz * dij * initstate.masses[j];
            accelerationsx[j] -= diffx * dij * initstate.masses[i];
			accelerationsy[j] -= diffy * dij * initstate.masses[i];
			accelerationsz[j] -= diffz * dij * initstate.masses[i];
			}
		}

	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}

//OMP  version
#elif STRATEGY == PARFOR_NAIVE
#pragma omp parallel for
    for (int i = 0; i < n_particles; i ++)
    {
        for (int j = 0; j < i; j++)
        {
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					__m256 x_vec = _mm256_set1_ps(dij);
					__m256 inv_sqrt_x_vec = _mm256_rsqrt_ps(x_vec);
					inv_sqrt_x_vec = _mm256_mul_ps(inv_sqrt_x_vec,_mm256_mul_ps(inv_sqrt_x_vec,inv_sqrt_x_vec));
					float inv_sqrt_x = _mm256_cvtss_f32(inv_sqrt_x_vec);
					dij = 10.0 * inv_sqrt_x;
				}
				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];

				accelerationsx[j] -= diffx * dij * initstate.masses[i];
				accelerationsy[j] -= diffy * dij * initstate.masses[i];
				accelerationsz[j] -= diffz * dij * initstate.masses[i];
			}
		}
	}
#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}

#elif STRATEGY == XSIMD

std::size_t inc = b_type::size;
std::size_t vec_size = n_particles - n_particles % inc;
    for (int i = 0; i < vec_size; i += inc)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
		b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
		b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
		std::vector<float> once(b_type::size,1.0);
		b_type once_v = b_type::load_unaligned(&once[0]);


        for(int j=0; j<n_particles; j ++){

			b_type diffx = particles.x[j] - rposx_i;
			b_type diffy = particles.y[j] - rposy_i;
			b_type diffz = particles.z[j] - rposz_i;	

			b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;	

			dij = xs::rsqrt(xs::max(once_v,dij)); //Il faut un batch de 1
			dij = 10.0 * dij * dij * dij;

			raccx_i = raccx_i + diffx * dij * initstate.masses[j];
			raccy_i = raccy_i + diffy * dij * initstate.masses[j];
			raccz_i = raccz_i + diffz * dij * initstate.masses[j];

			raccx_i.store_unaligned(&accelerationsx[i]);
			raccy_i.store_unaligned(&accelerationsy[i]);
			raccz_i.store_unaligned(&accelerationsz[i]);

		}
    }
	for (std::size_t i = vec_size; i<n_particles; i++)
	{
		for (int j = 0; j < i; j++)
        {
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					__m256 x_vec = _mm256_set1_ps(dij);
					__m256 inv_sqrt_x_vec = _mm256_rsqrt_ps(x_vec);
					inv_sqrt_x_vec = _mm256_mul_ps(inv_sqrt_x_vec,_mm256_mul_ps(inv_sqrt_x_vec,inv_sqrt_x_vec));
					float inv_sqrt_x = _mm256_cvtss_f32(inv_sqrt_x_vec);
					dij = 10.0 * inv_sqrt_x;
				}
				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];

				accelerationsx[j] -= diffx * dij * initstate.masses[i];
				accelerationsy[j] -= diffy * dij * initstate.masses[i];
				accelerationsz[j] -= diffz * dij * initstate.masses[i];
			}
		}

	}
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}

#elif STRATEGY == XSIMD_OMP
std::size_t inc = b_type::size;
std::size_t vec_size = n_particles - n_particles % inc;
#pragma omp parallel for simd
    for (int i = 0; i < vec_size; i += inc)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
		b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
		b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
		std::vector<float> once(b_type::size,1.0);
		b_type once_v = b_type::load_unaligned(&once[0]);


        for(int j=0; j<n_particles; j ++){

			b_type diffx = particles.x[j] - rposx_i;
			b_type diffy = particles.y[j] - rposy_i;
			b_type diffz = particles.z[j] - rposz_i;	

			b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;	
			
			dij = xs::rsqrt(xs::max(once_v,dij)); //Il faut un batch de 1
			dij = 10.0 * dij * dij * dij;
			if (!((j < i) | (j>i+7))) {
				b_type mask = b_type::load_unaligned(&masks[j-i][0]);
				dij = dij * mask;
			}

			raccx_i = raccx_i + diffx * dij * initstate.masses[j];
			raccy_i = raccy_i + diffy * dij * initstate.masses[j];
			raccz_i = raccz_i + diffz * dij * initstate.masses[j];

			raccx_i.store_unaligned(&accelerationsx[i]);
			raccy_i.store_unaligned(&accelerationsy[i]);
			raccz_i.store_unaligned(&accelerationsz[i]);
			
		}
    }

	#pragma omp parallel for
	for (std::size_t i = vec_size; i<n_particles; i++)
	{
		for (int j = 0; j < i; j++)
        {
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					__m256 x_vec = _mm256_set1_ps(dij);
					__m256 inv_sqrt_x_vec = _mm256_rsqrt_ps(x_vec);
					inv_sqrt_x_vec = _mm256_mul_ps(inv_sqrt_x_vec,_mm256_mul_ps(inv_sqrt_x_vec,inv_sqrt_x_vec));
					float inv_sqrt_x = _mm256_cvtss_f32(inv_sqrt_x_vec);
					dij = 10.0 * inv_sqrt_x;
				}
				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];

				accelerationsx[j] -= diffx * dij * initstate.masses[i];
				accelerationsy[j] -= diffy * dij * initstate.masses[i];
				accelerationsz[j] -= diffz * dij * initstate.masses[i];
			}
		}

	}
#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}


#elif STRATEGY == XSIMD_OMP_OPTI
std::size_t inc = b_type::size;
std::size_t vec_size = n_particles - n_particles % inc;
auto constexpr mask = xs::make_batch_constant<xs::batch<uint32_t, xs::avx2>, Rot>();

#pragma omp parallel for simd
    for (int i = 0; i < vec_size; i += inc)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
		b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
		b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
		std::vector<float> once(b_type::size,1.0);
		b_type once_v = b_type::load_unaligned(&once[0]);

        for(int j=0; j<vec_size; j += inc){
			if(i != j){
				b_type rposx_j = b_type::load_unaligned(&particles.x[j]);
				b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
				b_type rposz_j = b_type::load_unaligned(&particles.z[j]);
				b_type mass_j = b_type::load_unaligned(&initstate.masses[j]);
				for(int k = 0; k <7; k++){
					    
						b_type diffx = rposx_j - rposx_i;
						b_type diffy = rposy_j - rposy_i;
						b_type diffz = rposz_j - rposz_i;	
						rposx_j = xs::swizzle(rposx_j,mask);
						rposy_j = xs::swizzle(rposy_j,mask);
						rposz_j = xs::swizzle(rposz_j,mask);

						b_type dij = xs::fma(diffx,diffx,xs::fma(diffy,diffy, xs::mul(diffz,diffz)));	
						
						dij = xs::rsqrt(xs::max(once_v,dij)); //Il faut un batch de 1
						dij = 10.0 * dij * dij * dij;


						raccx_i = xs::fma(dij * mass_j, diffx,raccx_i);
						raccy_i = xs::fma(dij * mass_j, diffy,raccy_i);
						raccz_i = xs::fma(dij * mass_j, diffz,raccz_i);

						mass_j = xs::swizzle(mass_j,mask);
				}
				
				raccx_i.store_unaligned(&accelerationsx[i]);
				raccy_i.store_unaligned(&accelerationsy[i]);
				raccz_i.store_unaligned(&accelerationsz[i]);
				
			}
		}
		for (std::size_t j = vec_size; j<n_particles; j++)
		{
			b_type diffx = particles.x[j] - rposx_i;
			b_type diffy = particles.y[j] - rposy_i;
			b_type diffz = particles.z[j] - rposz_i;	

			b_type dij = xs::fma(diffx,diffx,xs::fma(diffy,diffy, xs::mul(diffz,diffz)));		
			
			dij = xs::rsqrt(xs::max(once_v,dij)); //Il faut un batch de 1
			dij = 10.0 * dij * dij * dij;
			if (!((j < i) | (j>i+7))) {
				b_type mask = b_type::load_unaligned(&masks[j-i][0]);
				dij = dij * mask;
			}

			raccx_i = xs::fma(dij * initstate.masses[i], diffx,raccx_i);
			raccy_i = xs::fma(dij * initstate.masses[i], diffy,raccy_i);
			raccz_i = xs::fma(dij * initstate.masses[i], diffz,raccz_i);

			raccx_i.store_unaligned(&accelerationsx[i]);
			raccy_i.store_unaligned(&accelerationsy[i]);
			raccz_i.store_unaligned(&accelerationsz[i]);
    	}	
	}

	#pragma omp parallel for
	for (std::size_t i = vec_size; i<n_particles; i++)
	{
		for (int j = 0; j < i; j++)
        {
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					__m256 x_vec = _mm256_set1_ps(dij);
					__m256 inv_sqrt_x_vec = _mm256_rsqrt_ps(x_vec);
					inv_sqrt_x_vec = _mm256_mul_ps(inv_sqrt_x_vec,_mm256_mul_ps(inv_sqrt_x_vec,inv_sqrt_x_vec));
					float inv_sqrt_x = _mm256_cvtss_f32(inv_sqrt_x_vec);
					dij = 10.0 * inv_sqrt_x;
				}
				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];

				accelerationsx[j] -= diffx * dij * initstate.masses[i];
				accelerationsy[j] -= diffy * dij * initstate.masses[i];
				accelerationsz[j] -= diffz * dij * initstate.masses[i];
			}
		}

	}
#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}


#endif
}

#endif //GALAX_MODEL_CPU_FAST
