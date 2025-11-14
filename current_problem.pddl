(define (problem tamp-iteration-1)
  (:domain blocksworld)

  (:objects
    b c g m r y - block
  )

  (:init
    (ontable r)
    (ontable g)
    (ontable b)
    (ontable y)
    (ontable m)
    (ontable c)
    (clear r)
    (clear g)
    (clear b)
    (clear y)
    (clear m)
    (clear c)
    (handempty)
  )

  (:goal (and
    (on r g)
    (on g b)
    (on y m)
    (on m c)
    (ontable b)
    (ontable c)
    (clear r)
    (clear y)
  ))
)
