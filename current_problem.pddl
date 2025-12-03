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
    (on c m)
    (on m y)
    (on y b)
    (on b r)
    (on r g)
    (ontable g)
    (clear c)
  ))
)
