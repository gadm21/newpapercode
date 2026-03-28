// ============================================
// WiFi Sensing Paper - Advanced Interactive Animations
// ============================================

class ParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.mouse = { x: null, y: null, radius: 150 };
        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouse.x = e.clientX - rect.left;
            this.mouse.y = e.clientY - rect.top;
        });
        this.canvas.addEventListener('mouseleave', () => {
            this.mouse.x = null;
            this.mouse.y = null;
        });
        this.createParticles();
        this.animate();
    }

    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
    }

    createParticles() {
        const particleCount = Math.floor((this.canvas.width * this.canvas.height) / 15000);
        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                size: Math.random() * 3 + 1,
                speedX: (Math.random() - 0.5) * 0.5,
                speedY: (Math.random() - 0.5) * 0.5,
                color: `hsla(${210 + Math.random() * 40}, 70%, 60%, ${0.3 + Math.random() * 0.4})`
            });
        }
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.particles.forEach((p, i) => {
            // Mouse interaction
            if (this.mouse.x !== null) {
                const dx = this.mouse.x - p.x;
                const dy = this.mouse.y - p.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < this.mouse.radius) {
                    const force = (this.mouse.radius - dist) / this.mouse.radius;
                    p.x -= dx * force * 0.03;
                    p.y -= dy * force * 0.03;
                }
            }

            // Update position
            p.x += p.speedX;
            p.y += p.speedY;

            // Wrap around
            if (p.x < 0) p.x = this.canvas.width;
            if (p.x > this.canvas.width) p.x = 0;
            if (p.y < 0) p.y = this.canvas.height;
            if (p.y > this.canvas.height) p.y = 0;

            // Draw particle
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color;
            this.ctx.fill();

            // Connect nearby particles
            for (let j = i + 1; j < this.particles.length; j++) {
                const p2 = this.particles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 100) {
                    this.ctx.beginPath();
                    this.ctx.strokeStyle = `rgba(100, 150, 200, ${0.2 * (1 - dist / 100)})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.moveTo(p.x, p.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.stroke();
                }
            }
        });

        requestAnimationFrame(() => this.animate());
    }
}

class MorphingText {
    constructor(element) {
        this.element = element;
        this.originalText = element.textContent;
        this.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        this.init();
    }

    init() {
        this.element.addEventListener('mouseenter', () => this.scramble());
    }

    scramble() {
        let iterations = 0;
        const interval = setInterval(() => {
            this.element.textContent = this.originalText
                .split('')
                .map((char, index) => {
                    if (index < iterations || char === ' ') return char;
                    return this.chars[Math.floor(Math.random() * this.chars.length)];
                })
                .join('');
            
            iterations += 1/3;
            if (iterations >= this.originalText.length) {
                clearInterval(interval);
                this.element.textContent = this.originalText;
            }
        }, 30);
    }
}

class MagneticButton {
    constructor(element) {
        this.element = element;
        this.boundingRect = null;
        this.init();
    }

    init() {
        this.element.style.transition = 'transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        
        this.element.addEventListener('mouseenter', () => {
            this.boundingRect = this.element.getBoundingClientRect();
        });

        this.element.addEventListener('mousemove', (e) => {
            if (!this.boundingRect) return;
            const x = e.clientX - this.boundingRect.left - this.boundingRect.width / 2;
            const y = e.clientY - this.boundingRect.top - this.boundingRect.height / 2;
            this.element.style.transform = `translate(${x * 0.3}px, ${y * 0.3}px) scale(1.05)`;
        });

        this.element.addEventListener('mouseleave', () => {
            this.element.style.transform = 'translate(0, 0) scale(1)';
            this.boundingRect = null;
        });
    }
}

class SmoothReveal {
    constructor() {
        this.elements = document.querySelectorAll('.animate-on-scroll');
        this.init();
    }

    init() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.classList.add('visible');
                        entry.target.style.transitionDelay = '0s';
                    }, index * 50);
                }
            });
        }, { threshold: 0.1, rootMargin: '-50px' });

        this.elements.forEach(el => observer.observe(el));
    }
}

class AnimatedCounter {
    constructor(element, target, duration = 2000) {
        this.element = element;
        this.target = target;
        this.duration = duration;
        this.hasAnimated = false;
    }

    animate() {
        if (this.hasAnimated) return;
        this.hasAnimated = true;

        const start = performance.now();
        const startValue = 0;

        const step = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / this.duration, 1);
            
            // Easing function (ease-out-expo)
            const easeProgress = 1 - Math.pow(2, -10 * progress);
            const currentValue = startValue + (this.target - startValue) * easeProgress;
            
            if (this.target % 1 === 0) {
                this.element.textContent = Math.floor(currentValue);
            } else {
                this.element.textContent = currentValue.toFixed(3);
            }

            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                this.element.textContent = this.target;
            }
        };

        requestAnimationFrame(step);
    }
}

class GlowingCursor {
    constructor() {
        this.cursor = document.createElement('div');
        this.cursor.className = 'glowing-cursor';
        document.body.appendChild(this.cursor);
        this.init();
    }

    init() {
        document.addEventListener('mousemove', (e) => {
            this.cursor.style.left = e.clientX + 'px';
            this.cursor.style.top = e.clientY + 'px';
        });

        document.querySelectorAll('a, button, .card, .branch, .pipeline-step, .figure-item, .layer-box').forEach(el => {
            el.addEventListener('mouseenter', () => {
                this.cursor.classList.add('cursor-hover');
            });
            el.addEventListener('mouseleave', () => {
                this.cursor.classList.remove('cursor-hover');
            });
        });
    }
}

class WaveAnimation {
    constructor(container) {
        this.container = container;
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'wave-canvas';
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');
        this.waves = [];
        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        for (let i = 0; i < 3; i++) {
            this.waves.push({
                amplitude: 20 + i * 10,
                frequency: 0.02 - i * 0.005,
                phase: i * Math.PI / 3,
                speed: 0.02 + i * 0.01,
                color: `rgba(72, 187, 120, ${0.1 + i * 0.05})`
            });
        }
        
        this.animate();
    }

    resize() {
        this.canvas.width = this.container.offsetWidth;
        this.canvas.height = 100;
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.waves.forEach(wave => {
            this.ctx.beginPath();
            this.ctx.moveTo(0, this.canvas.height / 2);
            
            for (let x = 0; x < this.canvas.width; x++) {
                const y = this.canvas.height / 2 + 
                    Math.sin(x * wave.frequency + wave.phase) * wave.amplitude;
                this.ctx.lineTo(x, y);
            }
            
            this.ctx.lineTo(this.canvas.width, this.canvas.height);
            this.ctx.lineTo(0, this.canvas.height);
            this.ctx.closePath();
            this.ctx.fillStyle = wave.color;
            this.ctx.fill();
            
            wave.phase += wave.speed;
        });
        
        requestAnimationFrame(() => this.animate());
    }
}

class DataFlowAnimation {
    constructor(container) {
        this.container = container;
        this.init();
    }

    init() {
        setInterval(() => this.createPacket(), 2000);
    }

    createPacket() {
        const packet = document.createElement('div');
        packet.className = 'data-packet';
        packet.innerHTML = '📦';
        this.container.appendChild(packet);
        
        setTimeout(() => packet.remove(), 3000);
    }
}

class Tilt3D {
    constructor(element) {
        this.element = element;
        this.init();
    }

    init() {
        this.element.style.transformStyle = 'preserve-3d';
        this.element.style.transition = 'transform 0.1s ease-out';

        this.element.addEventListener('mousemove', (e) => {
            const rect = this.element.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;
            
            this.element.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
        });

        this.element.addEventListener('mouseleave', () => {
            this.element.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)';
        });
    }
}

class TypewriterEffect {
    constructor(element, text, speed = 50) {
        this.element = element;
        this.text = text;
        this.speed = speed;
        this.index = 0;
    }

    start() {
        this.element.textContent = '';
        this.element.style.borderRight = '2px solid var(--accent-color)';
        this.type();
    }

    type() {
        if (this.index < this.text.length) {
            this.element.textContent += this.text.charAt(this.index);
            this.index++;
            setTimeout(() => this.type(), this.speed);
        } else {
            setTimeout(() => {
                this.element.style.borderRight = 'none';
            }, 500);
        }
    }
}

class RippleEffect {
    constructor(element) {
        this.element = element;
        this.init();
    }

    init() {
        this.element.style.position = 'relative';
        this.element.style.overflow = 'hidden';
        
        this.element.addEventListener('click', (e) => {
            const rect = this.element.getBoundingClientRect();
            const ripple = document.createElement('span');
            ripple.className = 'ripple';
            ripple.style.left = (e.clientX - rect.left) + 'px';
            ripple.style.top = (e.clientY - rect.top) + 'px';
            this.element.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
        });
    }
}

// ============================================
// Initialize Everything
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    // Particle background in header
    const headerCanvas = document.createElement('canvas');
    headerCanvas.className = 'particle-canvas';
    const header = document.querySelector('.paper-header');
    if (header) {
        header.style.position = 'relative';
        header.appendChild(headerCanvas);
        new ParticleSystem(headerCanvas);
    }

    // Glowing cursor
    new GlowingCursor();

    // Smooth reveal animations
    new SmoothReveal();

    // Morphing text on titles
    document.querySelectorAll('h2, h3').forEach(el => new MorphingText(el));

    // Magnetic buttons on cards
    document.querySelectorAll('.card, .branch, .finding').forEach(el => new MagneticButton(el));

    // 3D tilt on figures
    document.querySelectorAll('.figure-item, .table-container').forEach(el => new Tilt3D(el));

    // Ripple effect on clickable elements
    document.querySelectorAll('.pipeline-step, .branch, .card').forEach(el => new RippleEffect(el));

    // Animated counters for accuracy values
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const text = entry.target.textContent;
                const value = parseFloat(text);
                if (!isNaN(value) && value > 0 && value <= 1) {
                    const counter = new AnimatedCounter(entry.target, value, 1500);
                    counter.animate();
                }
            }
        });
    }, { threshold: 0.5 });

    document.querySelectorAll('.number.best').forEach(el => counterObserver.observe(el));

    // Bar chart animations
    initBarChartAnimations();

    // Image modal
    initImageModal();

    // Table interactions
    initTableAnimations();

    // Pipeline flow animation
    initPipelineAnimation();

    // CNN layer animation
    initCNNAnimation();

    // Scroll progress indicator
    initScrollProgress();

    // Parallax effects
    initParallax();
});

function initBarChartAnimations() {
    document.querySelectorAll('.animate-bar').forEach(bar => {
        bar.style.width = '0';
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.querySelectorAll('.animate-bar').forEach((bar, i) => {
                    setTimeout(() => {
                        bar.classList.add('visible');
                    }, i * 100);
                });
            }
        });
    }, { threshold: 0.3 });

    document.querySelectorAll('.chart-container').forEach(el => observer.observe(el));
}

function initImageModal() {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const captionText = document.getElementById('caption');
    const closeBtn = document.querySelector('.close');

    document.querySelectorAll('.zoomable').forEach(img => {
        img.addEventListener('click', function() {
            modal.style.display = 'flex';
            modal.style.alignItems = 'center';
            modal.style.justifyContent = 'center';
            modalImg.src = this.src;
            captionText.innerHTML = this.alt;
            document.body.style.overflow = 'hidden';
            
            // Add zoom animation
            modalImg.style.transform = 'scale(0.8)';
            modalImg.style.opacity = '0';
            setTimeout(() => {
                modalImg.style.transition = 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
                modalImg.style.transform = 'scale(1)';
                modalImg.style.opacity = '1';
            }, 50);
        });
    });

    const closeModal = () => {
        modalImg.style.transform = 'scale(0.8)';
        modalImg.style.opacity = '0';
        setTimeout(() => {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }, 300);
    };

    closeBtn?.addEventListener('click', closeModal);
    modal?.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

function initTableAnimations() {
    document.querySelectorAll('.data-table tbody tr').forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.01)';
            this.style.boxShadow = '0 4px 20px rgba(0,0,0,0.1)';
            this.style.zIndex = '10';
            this.style.position = 'relative';
        });
        
        row.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = 'none';
            this.style.zIndex = '1';
        });
    });

    document.querySelectorAll('.number.best').forEach(cell => {
        cell.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.3)';
            this.style.textShadow = '0 0 20px rgba(72, 187, 120, 0.8)';
        });
        cell.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.textShadow = 'none';
        });
    });
}

function initPipelineAnimation() {
    const steps = document.querySelectorAll('.pipeline-step');
    let currentStep = 0;

    const animateFlow = () => {
        steps.forEach((step, i) => {
            step.classList.remove('active-step');
            if (i === currentStep) {
                step.classList.add('active-step');
            }
        });
        currentStep = (currentStep + 1) % steps.length;
    };

    setInterval(animateFlow, 1500);

    steps.forEach((step, index) => {
        step.addEventListener('mouseenter', function() {
            steps.forEach((s, i) => {
                if (i <= index) {
                    s.style.background = 'linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%)';
                    s.style.borderColor = '#48bb78';
                    s.style.transform = 'translateY(-5px)';
                }
            });
        });
        
        step.addEventListener('mouseleave', function() {
            steps.forEach(s => {
                s.style.background = '';
                s.style.borderColor = '';
                s.style.transform = '';
            });
        });
    });
}

function initCNNAnimation() {
    const layers = document.querySelectorAll('.layer-box');
    
    layers.forEach((layer, i) => {
        layer.addEventListener('mouseenter', function() {
            // Highlight data flow
            layers.forEach((l, j) => {
                if (j <= i) {
                    l.style.boxShadow = '0 0 30px rgba(49, 130, 206, 0.5)';
                    l.style.transform = 'scale(1.05)';
                }
            });
        });
        
        layer.addEventListener('mouseleave', function() {
            layers.forEach(l => {
                l.style.boxShadow = '';
                l.style.transform = '';
            });
        });
    });
}

function initScrollProgress() {
    const progressBar = document.createElement('div');
    progressBar.className = 'scroll-progress';
    document.body.appendChild(progressBar);

    window.addEventListener('scroll', () => {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const progress = (scrollTop / docHeight) * 100;
        progressBar.style.width = progress + '%';
    });
}

function initParallax() {
    const parallaxElements = document.querySelectorAll('.paper-header, .conclusion');
    
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        
        parallaxElements.forEach(el => {
            const rate = scrolled * 0.3;
            el.style.backgroundPosition = `center ${rate}px`;
        });
    });
}

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});
